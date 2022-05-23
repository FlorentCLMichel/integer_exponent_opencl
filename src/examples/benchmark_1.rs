use integer_exponent_opencl::*;
use std::time::Instant;

fn main() {

    // parameters
    let m: usize = 10000; // number of integers
    let n: u32 = 400000; // power
    let q: u32 = 2022; // modulo

    // input
    let x: Vec<u32> = (0..(m as u32)).collect();

    // define the GPU context
    let context = define_context(0).unwrap();
        
    // define the GPU compute core
    let mut core = ExpModComp::<u32>::new("./src/exp_device.cl", m, &context).unwrap();
    
    // CPU benchmark
    let start_cpu = Instant::now();
    let y_cpu = exp_cpu(&x, n as usize, q);
    let n_millis_cpu = start_cpu.elapsed().as_millis();
    println!("Runtime on the CPU: {}ms", n_millis_cpu);

    // GPU benchmark
    let start_gpu = Instant::now();
    let y_gpu = core.compute(&x, n, q).unwrap();
    let n_millis_gpu = start_gpu.elapsed().as_millis();
    println!("Runtime on the GPU: {}ms", n_millis_gpu);

    // check the equality
    for i in 0..m {
        assert_eq!(y_cpu[i], y_gpu[i]);
    }
    println!("The results are equal");
}
