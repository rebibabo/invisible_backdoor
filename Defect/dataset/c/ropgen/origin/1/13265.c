void microblaze_load_kernel(MicroBlazeCPU *cpu, hwaddr ddr_base,

                            uint32_t ramsize,

                            const char *initrd_filename,

                            const char *dtb_filename,

                            void (*machine_cpu_reset)(MicroBlazeCPU *))

{

    QemuOpts *machine_opts;

    const char *kernel_filename;

    const char *kernel_cmdline;

    const char *dtb_arg;



    machine_opts = qemu_get_machine_opts();

    kernel_filename = qemu_opt_get(machine_opts, "kernel");

    kernel_cmdline = qemu_opt_get(machine_opts, "append");

    dtb_arg = qemu_opt_get(machine_opts, "dtb");

    if (dtb_arg) { /* Preference a -dtb argument */

        dtb_filename = dtb_arg;

    } else { /* default to pcbios dtb as passed by machine_init */

        dtb_filename = qemu_find_file(QEMU_FILE_TYPE_BIOS, dtb_filename);

    }



    boot_info.machine_cpu_reset = machine_cpu_reset;

    qemu_register_reset(main_cpu_reset, cpu);



    if (kernel_filename) {

        int kernel_size;

        uint64_t entry, low, high;

        uint32_t base32;

        int big_endian = 0;



#ifdef TARGET_WORDS_BIGENDIAN

        big_endian = 1;

#endif



        /* Boots a kernel elf binary.  */

        kernel_size = load_elf(kernel_filename, NULL, NULL,

                               &entry, &low, &high,

                               big_endian, ELF_MACHINE, 0);

        base32 = entry;

        if (base32 == 0xc0000000) {

            kernel_size = load_elf(kernel_filename, translate_kernel_address,

                                   NULL, &entry, NULL, NULL,

                                   big_endian, ELF_MACHINE, 0);

        }

        /* Always boot into physical ram.  */

        boot_info.bootstrap_pc = ddr_base + (entry & 0x0fffffff);



        /* If it wasn't an ELF image, try an u-boot image.  */

        if (kernel_size < 0) {

            hwaddr uentry, loadaddr;



            kernel_size = load_uimage(kernel_filename, &uentry, &loadaddr, 0);

            boot_info.bootstrap_pc = uentry;

            high = (loadaddr + kernel_size + 3) & ~3;

        }



        /* Not an ELF image nor an u-boot image, try a RAW image.  */

        if (kernel_size < 0) {

            kernel_size = load_image_targphys(kernel_filename, ddr_base,

                                              ram_size);

            boot_info.bootstrap_pc = ddr_base;

            high = (ddr_base + kernel_size + 3) & ~3;

        }



        if (initrd_filename) {

            int initrd_size;

            uint32_t initrd_offset;



            high = ROUND_UP(high + kernel_size, 4);

            boot_info.initrd_start = high;

            initrd_offset = boot_info.initrd_start - ddr_base;



            initrd_size = load_ramdisk(initrd_filename,

                                       boot_info.initrd_start,

                                       ram_size - initrd_offset);

            if (initrd_size < 0) {

                initrd_size = load_image_targphys(initrd_filename,

                                                  boot_info.initrd_start,

                                                  ram_size - initrd_offset);

            }

            if (initrd_size < 0) {

                error_report("qemu: could not load initrd '%s'\n",

                             initrd_filename);

                exit(EXIT_FAILURE);

            }

            boot_info.initrd_end = boot_info.initrd_start + initrd_size;

            high = ROUND_UP(high + initrd_size, 4);

        }



        boot_info.cmdline = high + 4096;

        if (kernel_cmdline && strlen(kernel_cmdline)) {

            pstrcpy_targphys("cmdline", boot_info.cmdline, 256, kernel_cmdline);

        }

        /* Provide a device-tree.  */

        boot_info.fdt = boot_info.cmdline + 4096;

        microblaze_load_dtb(boot_info.fdt, ram_size,

                            boot_info.initrd_start,

                            boot_info.initrd_end,

                            kernel_cmdline,

                            dtb_filename);

    }



}
