static void tc6393xb_gpio_handler_update(TC6393xbState *s)

{

    uint32_t level, diff;

    int bit;



    level = s->gpio_level & s->gpio_dir;



    for (diff = s->prev_level ^ level; diff; diff ^= 1 << bit) {

        bit = ffs(diff) - 1;

        qemu_set_irq(s->handler[bit], (level >> bit) & 1);

    }



    s->prev_level = level;

}
