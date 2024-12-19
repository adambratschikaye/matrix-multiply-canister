#![feature(simd_ffi)]

use std::cell::RefCell;

use candid::candid_method;
use ic_cdk::api::stable::{stable_grow, stable_read, stable_write};
use ic_cdk_macros::{init, update};

struct Data {
    a: Vec<i32>,
    b: Vec<i32>,
    out: Vec<i32>,
}

thread_local! {
    pub static DATA: RefCell<Data> =
      RefCell::new(Data {
        a: Vec::new(),
        b: Vec::new(),
        out: Vec::new()}
    );
}

#[cfg(target_arch = "wasm32")]
pub mod ic0 {
    #[link(wasm_import_module = "ic0")]
    extern "C" {
        pub fn stable_read_v128(src: u64) -> core::arch::wasm32::v128;
        pub fn stable_write_i32(dst: u64, val: i32);
    }
}

#[candid_method(init)]
#[init]
fn init(n: usize, d: usize) {
    DATA.with(|data| {
        let mut data = data.borrow_mut();
        data.a.reserve(n * d);
        for i in 0..n * d {
            data.a.push(i as u32 as i32);
        }

        data.b.reserve(n);
        for i in 0..n {
            data.b.push(i as u32 as i32);
        }

        data.out = vec![0; d];

        let stable_pages = ((n * d + n + d) * 4) / (64 * 1024) + 1;
        stable_grow(stable_pages as u64).unwrap();
        for i in 0..n * d {
            let val = data.a[i].to_le_bytes();
            stable_write((i * 4) as u64, &val);
        }
        for i in 0..n {
            let val = data.b[i].to_le_bytes();
            stable_write((n * d * 4 + i * 4) as u64, &val);
        }
    });
}

#[cfg(target_arch = "wasm32")]
#[candid_method(update)]
#[update]
pub fn multiply_stable() {
    use core::arch::wasm32::*;

    let (n, d) = DATA.with(|data| {
        let data = data.borrow();
        (data.b.len() as u64, data.out.len() as u64)
    });

    let a_addr = 0;
    let b_addr = n * d * 4;
    let out_addr = (n * d + n) * 4;

    for i in 0..d {
        let in_ = i * n * 4;
        let mut vals = i32x4(0, 0, 0, 0);
        for j in (0..n).step_by(4) {
            let a_group: v128 = unsafe { ic0::stable_read_v128(a_addr + in_ + j * 4) };
            let b_group: v128 = unsafe { ic0::stable_read_v128(b_addr + j * 4) };
            vals = i32x4_add(vals, i32x4_mul(a_group, b_group));
        }
        let val = i32x4_extract_lane::<0>(vals)
            + i32x4_extract_lane::<1>(vals)
            + i32x4_extract_lane::<2>(vals)
            + i32x4_extract_lane::<3>(vals);
        unsafe { ic0::stable_write_i32(out_addr + i * 4, val) };
    }
}

#[candid_method(update)]
#[update]
pub fn multiply_stable_old() {
    let (n, d) = DATA.with(|data| {
        let data = data.borrow();
        (data.b.len() as u64, data.out.len() as u64)
    });

    let a_addr = 0;
    let b_addr = n * d * 4;
    let out_addr = (n * d + n) * 4;
    // let mut a_group = Box::new(i32x4(0, 0, 0, 0));
    // let mut b_group = Box::new(i32x4(0, 0, 0, 0));
    let mut a_group = [0_u8; 16];
    let mut b_group = [0_u8; 16];

    for i in 0..d {
        let in_ = i * n * 4;
        let mut val = 0;
        for j in (0..n).step_by(4) {
            stable_read(a_addr + in_ + j * 4, &mut a_group);
            stable_read(b_addr + j * 4, &mut b_group);
            let a_group: [i32; 4] = unsafe { std::mem::transmute(a_group) };
            let b_group: [i32; 4] = unsafe { std::mem::transmute(b_group) };
            let a_group = a_group.as_ptr();
            let b_group = b_group.as_ptr();

            let mut ival: i32 = 0;
            for i in 0..4 {
                ival += unsafe { *a_group.add(i) * *b_group.add(i) };
            }

            val += ival;
        }
        stable_write(out_addr + i * 4, &val.to_le_bytes());
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[candid_method(update)]
#[update]
pub fn multiply_stable() {}

#[candid_method(update)]
#[update]
pub fn update_empty() {}

pub fn matmul<const GROUP_SIZE: usize>() {
    DATA.with(|data| {
        let mut data = data.borrow_mut();

        let n = data.b.len();
        let d = data.out.len();

        assert_eq!(data.a.len(), n * d);

        for i in 0..d {
            let in_ = i * n;
            let mut val = 0;

            // matmul in groups of `GROUP_SIZE`.
            for j in (0..n).step_by(GROUP_SIZE) {
                // NOTE: Using raw pointer arithmetic as it reduces the overhead of bound checks.
                // Expeirments showed that it reduces instructions by 6.6%.
                unsafe {
                    let b_group = data.b.as_ptr().add(j);
                    let a_group = data.a.as_ptr().add(in_ + j);

                    // multiply and sum both groups.
                    let mut ival: i32 = 0;
                    for i in 0..GROUP_SIZE {
                        ival += *a_group.add(i) as i32 * *b_group.add(i) as i32;
                    }

                    val += ival;
                }
            }
            unsafe { *data.out.as_mut_ptr().add(i) = val };
        }
    });
}

#[candid_method(update)]
#[update]
fn multiply_heap() {
    matmul::<64>();
}

// When run on native this prints the candid service definition of this
// canister, from the methods annotated with `candid_method` above.
//
// Note that `cargo test` calls `main`, and `export_service` (which defines
// `__export_service` in the current scope) needs to be called exactly once. So
// in addition to `not(target_family = "wasm")` we have a `not(test)` guard here
// to avoid calling `export_service`, which we need to call in the test below.
#[cfg(not(any(target_family = "wasm", test)))]
fn main() {
    // The line below generates did types and service definition from the
    // methods annotated with `candid_method` above. The definition is then
    // obtained with `__export_service()`.
    candid::export_service!();
    std::print!("{}", __export_service());
}

#[cfg(any(target_family = "wasm", test))]
fn main() {}

#[test]
fn check_candid_file() {
    let did_path = match std::env::var("DID_PATH") {
        Ok(v) => v,
        Err(_e) => "matrix-multiply.did".to_string(),
    };
    let candid = String::from_utf8(std::fs::read(did_path).unwrap()).unwrap();

    // See comments in main above
    candid::export_service!();
    let expected = __export_service();

    assert_eq!(candid, expected);
}
