use opencl3::context::Context;
use opencl3::kernel::Kernel;
use opencl3::device::{ Device, CL_DEVICE_TYPE_GPU };
use opencl3::program::{ Program, CL_STD_2_0 };
use opencl3::command_queue::{ CommandQueue, CL_QUEUE_PROFILING_ENABLE };

const KERNEL_NAME: &str = "exp_device";

pub fn initialize_context_kernel(source_file: &str) 
    -> (Context, Kernel)
{
    // Find a usable platform and device for this application
    let platforms = opencl3::platform::get_platforms()?;
    let platform = platforms.first().expect("no OpenCL platforms");
    let device = *platform
        .get_devices(CL_DEVICE_TYPE_GPU)?
        .first()
        .expect("no device found in platform");
    let device = Device::new(device);
    
    // Create a Context on an OpenCL device
    let context = Context::from_device(&device).expect("Context::from_device failed");

    // read the program source
    let program_source = std::fs::read_all(source_file)
        .expect(&format!("Could not read the source file {}", source_file));

    // Build the OpenCL program source and create the kernel.
    let program = Program::create_and_build_from_source(&context, program_source, CL_STD_2_0)
        .expect("Program::create_and_build_from_source failed");
    let kernel = Kernel::create(&program, KERNEL_NAME).expect("Kernel::create failed");

    // Create a command_queue on the Context's device
    let queue = CommandQueue::create_with_properties(
        &context,
        context.default_device(),
        CL_QUEUE_PROFILING_ENABLE,
        0,
    )
    .expect("CommandQueue::create_with_properties failed");

    (context, kernel)
}
