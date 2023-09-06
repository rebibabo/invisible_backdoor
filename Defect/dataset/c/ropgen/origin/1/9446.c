void serial_exit_core(SerialState *s)
{
    qemu_chr_fe_deinit(&s->chr);
    qemu_unregister_reset(serial_reset, s);
}