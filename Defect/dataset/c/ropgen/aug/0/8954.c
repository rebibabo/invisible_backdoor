static void dummy_m68k_init(QEMUMachineInitArgs *args)

{

    ram_addr_t ram_size = args->ram_size;

    const char *cpu_model = args->cpu_model;

    const char *kernel_filename = args->kernel_filename;

    CPUM68KState *env;

    MemoryRegion *address_space_mem =  get_system_memory();

    MemoryRegion *ram = g_new(MemoryRegion, 1);

    int kernel_size;

    uint64_t elf_entry;

    target_phys_addr_t entry;



    if (!cpu_model)

        cpu_model = "cfv4e";

    env = cpu_init(cpu_model);

    if (!env) {

        fprintf(stderr, "Unable to find m68k CPU definition\n");

        exit(1);

    }



    /* Initialize CPU registers.  */

    env->vbr = 0;



    /* RAM at address zero */

    memory_region_init_ram(ram, "dummy_m68k.ram", ram_size);

    vmstate_register_ram_global(ram);

    memory_region_add_subregion(address_space_mem, 0, ram);



    /* Load kernel.  */

    if (kernel_filename) {

        kernel_size = load_elf(kernel_filename, NULL, NULL, &elf_entry,

                               NULL, NULL, 1, ELF_MACHINE, 0);

        entry = elf_entry;

        if (kernel_size < 0) {

            kernel_size = load_uimage(kernel_filename, &entry, NULL, NULL);

        }

        if (kernel_size < 0) {

            kernel_size = load_image_targphys(kernel_filename,

                                              KERNEL_LOAD_ADDR,

                                              ram_size - KERNEL_LOAD_ADDR);

            entry = KERNEL_LOAD_ADDR;

        }

        if (kernel_size < 0) {

            fprintf(stderr, "qemu: could not load kernel '%s'\n",

                    kernel_filename);

            exit(1);

        }

    } else {

        entry = 0;

    }

    env->pc = entry;

}
