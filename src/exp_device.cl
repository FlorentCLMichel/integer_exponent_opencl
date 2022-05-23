typedef uint NUMBER; // 32-bit insigned integers

kernel void exp_device (global unsigned int* output, 
                        global const unsigned int* input,
                        NUMBER n,
                        NUMBER q)
{
    size_t lid = get_local_id(0);
    size_t lsize = get_local_size(0);
    size_t num_groups = get_num_groups(0);
    
    for (size_t i = 0; i < num_groups; ++i) {
        size_t lidx = i * lsize + lid;
        NUMBER x = input[lidx];
        NUMBER m = 1;
        for (NUMBER j = 0; j < n; ++j) {
            m = (m*x) % q;
        }
        output[lidx] = m;
    }
}
