use opencl3::context::Context;
use opencl3::device::{ Device, CL_DEVICE_TYPE_GPU };
use opencl3::program::{ Program, CL_STD_2_0 };
use opencl3::command_queue::{ CommandQueue, CL_QUEUE_PROFILING_ENABLE };
use opencl3::error_codes::ClError;
use opencl3::svm::SvmVec;
use opencl3::kernel::{ ExecuteKernel, Kernel };
use opencl3::memory::{ CL_MAP_READ, CL_MAP_WRITE };
use opencl3::types::CL_BLOCKING;
use crate:: Number;

const KERNEL_NAME: &str = "exp_device";

pub struct ExpModComp<'a, T: Number> {
    n_elements: usize,
    kernel: Kernel, 
    queue: CommandQueue,
    input_vec: SvmVec<'a, T>,
    output_vec: SvmVec<'a, T>,
}


pub fn define_context() -> Result<Context, ExpModError>
{
    // Find a usable platform and device for this application
    let platforms = opencl3::platform::get_platforms()?;
    let platform = platforms.first()
                            .ok_or(ExpModError { message: "No platform found".to_string() })?;
    let device = *platform
        .get_devices(CL_DEVICE_TYPE_GPU)?
        .first()
        .ok_or(ExpModError { message: "No GPU found".to_string() })?;
    let device = Device::new(device);
    
    // Create a Context on an OpenCL device
    Ok(Context::from_device(&device)?)
}


impl<'a, T: Number> ExpModComp<'a, T> {
    pub fn new(source_file: &str, n_elements: usize, context: &'a Context) 
        -> Result<Self, ExpModError>
    {
    
        // read the program source
        let program_source = std::fs::read_to_string(source_file)?;
    
        // Build the OpenCL program source and create the kernel.
        let program = Program::create_and_build_from_source(context, &program_source, CL_STD_2_0)?;
        let kernel = Kernel::create(&program, KERNEL_NAME)?;
    
        // Create a command_queue on the Context's device
        let queue = CommandQueue::create_with_properties(
            context,
            context.default_device(),
            CL_QUEUE_PROFILING_ENABLE,
            0,
        )?;

        // input and output SVM vectors
        let input_vec = SvmVec::<T>::allocate(context, n_elements)?;
        let output_vec = SvmVec::<T>::allocate(context, n_elements)?;
    
        Ok(ExpModComp { n_elements, kernel, queue, input_vec, output_vec })
    }

    pub fn compute(&mut self, x: &[T], n: T, q: T) -> Result<Vec<T>, ExpModError> {
        
        // check that `x` has the right length
        if x.len() != self.n_elements {
            return Err(ExpModError { 
                message: format!("Invalid array length: expected {}, got {}", 
                                 self.n_elements, x.len())
            });
        }

        // map the input array if not CL_MEM_SVM_FINE_GRAIN_BUFFER
        if !self.input_vec.is_fine_grained() {
            self.queue.enqueue_svm_map(CL_BLOCKING, CL_MAP_WRITE, &mut self.input_vec, &[]);
        }

        // copy the input to the SVN vector
        self.input_vec.clone_from_slice(x);
        
        // unmap the input array if not CL_MEM_SVM_FINE_GRAIN_BUFFER
        if !self.input_vec.is_fine_grained() {
            let unmap_event = self.queue.enqueue_svm_unmap(&mut self.input_vec, &[])?;
            unmap_event.wait();
        }

        // run the calculation
        let kernel_event = ExecuteKernel::new(&self.kernel)
            .set_arg_svm(self.output_vec.as_mut_ptr())
            .set_arg_svm(self.input_vec.as_ptr())
            .set_arg(&n)
            .set_arg(&q)
            .set_global_work_size(self.n_elements)
            .enqueue_nd_range(&self.queue)?;
        kernel_event.wait()?;

        // map the output array if not CL_MEM_SVM_FINE_GRAIN_BUFFER
        if !self.output_vec.is_fine_grained() {
            self.queue.enqueue_svm_map(CL_BLOCKING, CL_MAP_READ, &mut self.output_vec, &[]);
        }

        // Read the results
        let mut y = vec![T::one(); self.n_elements];
        y.clone_from_slice(&self.output_vec);
        
        // unmap the output array if not CL_MEM_SVM_FINE_GRAIN_BUFFER
        if !self.output_vec.is_fine_grained() {
            let unmap_event = self.queue.enqueue_svm_unmap(&mut self.output_vec, &[])?;
            unmap_event.wait();
        }

        Ok(y)
    }
}


/// A custom error type
#[derive(Clone, Debug)]
pub struct ExpModError { message: String }

impl std::fmt::Display for ExpModError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result 
    {
        write!(f, "{}", &self.message)
    }
}

impl std::convert::From<ClError> for ExpModError {
    fn from(err: ClError) -> Self {
        ExpModError { message: format!("ClERror: {}", err) }
    }
}

impl std::convert::From<String> for ExpModError {
    fn from(err: String) -> Self {
        ExpModError { message: err }
    }
}

impl std::convert::From<std::io::Error> for ExpModError {
    fn from(err: std::io::Error) -> Self {
        ExpModError { message: format!("IO Error: {}", err) }
    }
}

impl std::error::Error for ExpModError {}



#[cfg(test)]
mod tests {
    
    use super::*;

    #[test]
    fn build_context() {
        let context = define_context();
    }
    
    #[test]
    fn build_expmodcomp() {
        let n_elements: usize = 10;
        let context = define_context().unwrap();
        let core = ExpModComp::<u32>::new("./src/exp_device.cl", n_elements, &context).unwrap();
    }
    
    #[test]
    fn expmodcomp_0() {
        let n_elements: usize = 10;
        let context = define_context().unwrap();
        let mut core = ExpModComp::<u32>::new("./src/exp_device.cl", n_elements, &context).unwrap();
        let n: u32 = 0;
        let q: u32 = 10;
        let x: Vec<u32> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let y = core.compute(&x, n, q).unwrap();
        let exp_x = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        for i in 0..x.len() {
            assert_eq!(y[i], exp_x[i]);
        }
    }
    
    #[test]
    fn expmodcomp_1() {
        let n_elements: usize = 10;
        let context = define_context().unwrap();
        let mut core = ExpModComp::<u32>::new("./src/exp_device.cl", n_elements, &context).unwrap();
        let n: u32 = 1;
        let q: u32 = 10;
        let x: Vec<u32> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let y = core.compute(&x, n, q).unwrap();
        let exp_x = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        for i in 0..x.len() {
            assert_eq!(y[i], exp_x[i]);
        }
    }
    
    #[test]
    fn expmodcomp_2() {
        let n_elements: usize = 10;
        let context = define_context().unwrap();
        let mut core = ExpModComp::<u32>::new("./src/exp_device.cl", n_elements, &context).unwrap();
        let n: u32 = 2;
        let q: u32 = 10;
        let x: Vec<u32> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let y = core.compute(&x, n, q).unwrap();
        let exp_x = vec![0, 1, 4, 9, 6, 5, 6, 9, 4, 1];
        for i in 0..x.len() {
            assert_eq!(y[i], exp_x[i]);
        }
    }
}
