static void ppc_core99_init(QEMUMachineInitArgs *args)

{

    ram_addr_t ram_size = args->ram_size;

    const char *cpu_model = args->cpu_model;

    const char *kernel_filename = args->kernel_filename;

    const char *kernel_cmdline = args->kernel_cmdline;

    const char *initrd_filename = args->initrd_filename;

    const char *boot_device = args->boot_device;

    PowerPCCPU *cpu = NULL;

    CPUPPCState *env = NULL;

    char *filename;

    qemu_irq *pic, **openpic_irqs;

    MemoryRegion *unin_memory = g_new(MemoryRegion, 1);

    int linux_boot, i;

    MemoryRegion *ram = g_new(MemoryRegion, 1), *bios = g_new(MemoryRegion, 1);

    hwaddr kernel_base, initrd_base, cmdline_base = 0;

    long kernel_size, initrd_size;

    PCIBus *pci_bus;

    MacIONVRAMState *nvr;

    int bios_size;

    MemoryRegion *pic_mem, *dbdma_mem, *cuda_mem, *escc_mem;

    MemoryRegion *escc_bar = g_new(MemoryRegion, 1);

    MemoryRegion *ide_mem[3];

    int ppc_boot_device;

    DriveInfo *hd[MAX_IDE_BUS * MAX_IDE_DEVS];

    void *fw_cfg;

    void *dbdma;

    int machine_arch;



    linux_boot = (kernel_filename != NULL);



    /* init CPUs */

    if (cpu_model == NULL)

#ifdef TARGET_PPC64

        cpu_model = "970fx";

#else

        cpu_model = "G4";

#endif

    for (i = 0; i < smp_cpus; i++) {

        cpu = cpu_ppc_init(cpu_model);

        if (cpu == NULL) {

            fprintf(stderr, "Unable to find PowerPC CPU definition\n");

            exit(1);

        }

        env = &cpu->env;



        /* Set time-base frequency to 100 Mhz */

        cpu_ppc_tb_init(env, 100UL * 1000UL * 1000UL);

        qemu_register_reset(ppc_core99_reset, cpu);

    }



    /* allocate RAM */

    memory_region_init_ram(ram, "ppc_core99.ram", ram_size);

    vmstate_register_ram_global(ram);

    memory_region_add_subregion(get_system_memory(), 0, ram);



    /* allocate and load BIOS */

    memory_region_init_ram(bios, "ppc_core99.bios", BIOS_SIZE);

    vmstate_register_ram_global(bios);

    if (bios_name == NULL)

        bios_name = PROM_FILENAME;

    filename = qemu_find_file(QEMU_FILE_TYPE_BIOS, bios_name);

    memory_region_set_readonly(bios, true);

    memory_region_add_subregion(get_system_memory(), PROM_ADDR, bios);



    /* Load OpenBIOS (ELF) */

    if (filename) {

        bios_size = load_elf(filename, NULL, NULL, NULL,

                             NULL, NULL, 1, ELF_MACHINE, 0);



        g_free(filename);

    } else {

        bios_size = -1;

    }

    if (bios_size < 0 || bios_size > BIOS_SIZE) {

        hw_error("qemu: could not load PowerPC bios '%s'\n", bios_name);

        exit(1);

    }



    if (linux_boot) {

        uint64_t lowaddr = 0;

        int bswap_needed;



#ifdef BSWAP_NEEDED

        bswap_needed = 1;

#else

        bswap_needed = 0;

#endif

        kernel_base = KERNEL_LOAD_ADDR;



        kernel_size = load_elf(kernel_filename, translate_kernel_address, NULL,

                               NULL, &lowaddr, NULL, 1, ELF_MACHINE, 0);

        if (kernel_size < 0)

            kernel_size = load_aout(kernel_filename, kernel_base,

                                    ram_size - kernel_base, bswap_needed,

                                    TARGET_PAGE_SIZE);

        if (kernel_size < 0)

            kernel_size = load_image_targphys(kernel_filename,

                                              kernel_base,

                                              ram_size - kernel_base);

        if (kernel_size < 0) {

            hw_error("qemu: could not load kernel '%s'\n", kernel_filename);

            exit(1);

        }

        /* load initrd */

        if (initrd_filename) {

            initrd_base = round_page(kernel_base + kernel_size + KERNEL_GAP);

            initrd_size = load_image_targphys(initrd_filename, initrd_base,

                                              ram_size - initrd_base);

            if (initrd_size < 0) {

                hw_error("qemu: could not load initial ram disk '%s'\n",

                         initrd_filename);

                exit(1);

            }

            cmdline_base = round_page(initrd_base + initrd_size);

        } else {

            initrd_base = 0;

            initrd_size = 0;

            cmdline_base = round_page(kernel_base + kernel_size + KERNEL_GAP);

        }

        ppc_boot_device = 'm';

    } else {

        kernel_base = 0;

        kernel_size = 0;

        initrd_base = 0;

        initrd_size = 0;

        ppc_boot_device = '\0';

        /* We consider that NewWorld PowerMac never have any floppy drive

         * For now, OHW cannot boot from the network.

         */

        for (i = 0; boot_device[i] != '\0'; i++) {

            if (boot_device[i] >= 'c' && boot_device[i] <= 'f') {

                ppc_boot_device = boot_device[i];

                break;

            }

        }

        if (ppc_boot_device == '\0') {

            fprintf(stderr, "No valid boot device for Mac99 machine\n");

            exit(1);

        }

    }



    /* Register 8 MB of ISA IO space */

    isa_mmio_init(0xf2000000, 0x00800000);



    /* UniN init */

    memory_region_init_io(unin_memory, &unin_ops, NULL, "unin", 0x1000);

    memory_region_add_subregion(get_system_memory(), 0xf8000000, unin_memory);



    openpic_irqs = g_malloc0(smp_cpus * sizeof(qemu_irq *));

    openpic_irqs[0] =

        g_malloc0(smp_cpus * sizeof(qemu_irq) * OPENPIC_OUTPUT_NB);

    for (i = 0; i < smp_cpus; i++) {

        /* Mac99 IRQ connection between OpenPIC outputs pins

         * and PowerPC input pins

         */

        switch (PPC_INPUT(env)) {

        case PPC_FLAGS_INPUT_6xx:

            openpic_irqs[i] = openpic_irqs[0] + (i * OPENPIC_OUTPUT_NB);

            openpic_irqs[i][OPENPIC_OUTPUT_INT] =

                ((qemu_irq *)env->irq_inputs)[PPC6xx_INPUT_INT];

            openpic_irqs[i][OPENPIC_OUTPUT_CINT] =

                ((qemu_irq *)env->irq_inputs)[PPC6xx_INPUT_INT];

            openpic_irqs[i][OPENPIC_OUTPUT_MCK] =

                ((qemu_irq *)env->irq_inputs)[PPC6xx_INPUT_MCP];

            /* Not connected ? */

            openpic_irqs[i][OPENPIC_OUTPUT_DEBUG] = NULL;

            /* Check this */

            openpic_irqs[i][OPENPIC_OUTPUT_RESET] =

                ((qemu_irq *)env->irq_inputs)[PPC6xx_INPUT_HRESET];

            break;

#if defined(TARGET_PPC64)

        case PPC_FLAGS_INPUT_970:

            openpic_irqs[i] = openpic_irqs[0] + (i * OPENPIC_OUTPUT_NB);

            openpic_irqs[i][OPENPIC_OUTPUT_INT] =

                ((qemu_irq *)env->irq_inputs)[PPC970_INPUT_INT];

            openpic_irqs[i][OPENPIC_OUTPUT_CINT] =

                ((qemu_irq *)env->irq_inputs)[PPC970_INPUT_INT];

            openpic_irqs[i][OPENPIC_OUTPUT_MCK] =

                ((qemu_irq *)env->irq_inputs)[PPC970_INPUT_MCP];

            /* Not connected ? */

            openpic_irqs[i][OPENPIC_OUTPUT_DEBUG] = NULL;

            /* Check this */

            openpic_irqs[i][OPENPIC_OUTPUT_RESET] =

                ((qemu_irq *)env->irq_inputs)[PPC970_INPUT_HRESET];

            break;

#endif /* defined(TARGET_PPC64) */

        default:

            hw_error("Bus model not supported on mac99 machine\n");

            exit(1);

        }

    }

    pic = openpic_init(&pic_mem, smp_cpus, openpic_irqs, NULL);

    if (PPC_INPUT(env) == PPC_FLAGS_INPUT_970) {

        /* 970 gets a U3 bus */

        pci_bus = pci_pmac_u3_init(pic, get_system_memory(), get_system_io());

        machine_arch = ARCH_MAC99_U3;

    } else {

        pci_bus = pci_pmac_init(pic, get_system_memory(), get_system_io());

        machine_arch = ARCH_MAC99;

    }

    /* init basic PC hardware */

    pci_vga_init(pci_bus);



    escc_mem = escc_init(0, pic[0x25], pic[0x24],

                         serial_hds[0], serial_hds[1], ESCC_CLOCK, 4);

    memory_region_init_alias(escc_bar, "escc-bar",

                             escc_mem, 0, memory_region_size(escc_mem));



    for(i = 0; i < nb_nics; i++)

        pci_nic_init_nofail(&nd_table[i], "ne2k_pci", NULL);



    ide_drive_get(hd, MAX_IDE_BUS);

    dbdma = DBDMA_init(&dbdma_mem);



    /* We only emulate 2 out of 3 IDE controllers for now */

    ide_mem[0] = NULL;

    ide_mem[1] = pmac_ide_init(hd, pic[0x0d], dbdma, 0x16, pic[0x02]);

    ide_mem[2] = pmac_ide_init(&hd[MAX_IDE_DEVS], pic[0x0e], dbdma, 0x1a, pic[0x02]);



    cuda_init(&cuda_mem, pic[0x19]);



    adb_kbd_init(&adb_bus);

    adb_mouse_init(&adb_bus);



    macio_init(pci_bus, PCI_DEVICE_ID_APPLE_UNI_N_KEYL, 0, pic_mem,

               dbdma_mem, cuda_mem, NULL, 3, ide_mem, escc_bar);



    if (usb_enabled(machine_arch == ARCH_MAC99_U3)) {

        pci_create_simple(pci_bus, -1, "pci-ohci");

        /* U3 needs to use USB for input because Linux doesn't support via-cuda

        on PPC64 */

        if (machine_arch == ARCH_MAC99_U3) {

            usbdevice_create("keyboard");

            usbdevice_create("mouse");

        }

    }



    if (graphic_depth != 15 && graphic_depth != 32 && graphic_depth != 8)

        graphic_depth = 15;



    /* The NewWorld NVRAM is not located in the MacIO device */

    nvr = macio_nvram_init(0x2000, 1);

    pmac_format_nvram_partition(nvr, 0x2000);

    macio_nvram_setup_bar(nvr, get_system_memory(), 0xFFF04000);

    /* No PCI init: the BIOS will do it */



    fw_cfg = fw_cfg_init(0, 0, CFG_ADDR, CFG_ADDR + 2);

    fw_cfg_add_i32(fw_cfg, FW_CFG_ID, 1);

    fw_cfg_add_i64(fw_cfg, FW_CFG_RAM_SIZE, (uint64_t)ram_size);

    fw_cfg_add_i16(fw_cfg, FW_CFG_MACHINE_ID, machine_arch);

    fw_cfg_add_i32(fw_cfg, FW_CFG_KERNEL_ADDR, kernel_base);

    fw_cfg_add_i32(fw_cfg, FW_CFG_KERNEL_SIZE, kernel_size);

    if (kernel_cmdline) {

        fw_cfg_add_i32(fw_cfg, FW_CFG_KERNEL_CMDLINE, cmdline_base);

        pstrcpy_targphys("cmdline", cmdline_base, TARGET_PAGE_SIZE, kernel_cmdline);

    } else {

        fw_cfg_add_i32(fw_cfg, FW_CFG_KERNEL_CMDLINE, 0);

    }

    fw_cfg_add_i32(fw_cfg, FW_CFG_INITRD_ADDR, initrd_base);

    fw_cfg_add_i32(fw_cfg, FW_CFG_INITRD_SIZE, initrd_size);

    fw_cfg_add_i16(fw_cfg, FW_CFG_BOOT_DEVICE, ppc_boot_device);



    fw_cfg_add_i16(fw_cfg, FW_CFG_PPC_WIDTH, graphic_width);

    fw_cfg_add_i16(fw_cfg, FW_CFG_PPC_HEIGHT, graphic_height);

    fw_cfg_add_i16(fw_cfg, FW_CFG_PPC_DEPTH, graphic_depth);



    fw_cfg_add_i32(fw_cfg, FW_CFG_PPC_IS_KVM, kvm_enabled());

    if (kvm_enabled()) {

#ifdef CONFIG_KVM

        uint8_t *hypercall;



        fw_cfg_add_i32(fw_cfg, FW_CFG_PPC_TBFREQ, kvmppc_get_tbfreq());

        hypercall = g_malloc(16);

        kvmppc_get_hypercall(env, hypercall, 16);

        fw_cfg_add_bytes(fw_cfg, FW_CFG_PPC_KVM_HC, hypercall, 16);

        fw_cfg_add_i32(fw_cfg, FW_CFG_PPC_KVM_PID, getpid());

#endif

    } else {

        fw_cfg_add_i32(fw_cfg, FW_CFG_PPC_TBFREQ, get_ticks_per_sec());

    }



    qemu_register_boot_set(fw_cfg_boot_set, fw_cfg);

}
