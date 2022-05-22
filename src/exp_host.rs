use std::ops;
use num::traits::One;
use rayon::prelude::*;

/// a trait for acceptable number types
pub trait Number: Copy + ops::Mul + ops::Rem<Output=Self> + One + Sync + Send {}

/// Take each element of `x` to the power `n` modulo `q` on the CPU
pub fn exp_cpu<N: Number>(x: &[N], n: usize, q: N) -> Vec<N>
{
    x.par_iter()
     .map(|&e| exp_modulo::<N>(e, n, q))
     .collect()
}

fn exp_modulo<N: Number>(e: N, n: usize, q: N) -> N 
{
    let mut f = N::one();
    for _ in 0..n {
        f = (f * e) % q;
    }
    f
}

// implement the trait umber for common integer types
impl Number for u8 {}
impl Number for u16 {}
impl Number for u32 {}
impl Number for u64 {}
impl Number for u128 {}
impl Number for usize {}
impl Number for i8 {}
impl Number for i16 {}
impl Number for i32 {}
impl Number for i64 {}
impl Number for i128 {}
impl Number for isize {}


#[cfg(test)]
mod tests {
    
    use super::*;
    
    #[test]
    fn exp_cpu_0() {
        let x: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
        let q = 10;
        let n = 0;
        let y = exp_cpu(&x, n, q);
        let exp_x = vec![1, 1, 1, 1, 1, 1];
        for (a, b) in y.iter().zip(exp_x.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn exp_cpu_1() {
        let x: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
        let q = 10;
        let n = 1;
        let y = exp_cpu(&x, n, q);
        let exp_x = vec![1, 2, 3, 4, 5, 6];
        for (a, b) in y.iter().zip(exp_x.iter()) {
            assert_eq!(a, b);
        }
    }
    
    #[test]
    fn exp_cpu_2() {
        let x: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
        let q = 10;
        let n = 2;
        let y = exp_cpu(&x, n, q);
        let exp_x = vec![1, 4, 9, 6, 5, 6];
        for (a, b) in y.iter().zip(exp_x.iter()) {
            assert_eq!(a, b);
        }
    }
    
    #[test]
    fn exp_cpu_3() {
        let x: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
        let q = 10;
        let n = 3;
        let y = exp_cpu(&x, n, q);
        let exp_x = vec![1, 8, 7, 4, 5, 6];
        for (a, b) in y.iter().zip(exp_x.iter()) {
            assert_eq!(a, b);
        }
    }
}
