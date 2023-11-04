static void rocker_io_writel(void *opaque, hwaddr addr, uint32_t val)

{

    Rocker *r = opaque;



    if (rocker_addr_is_desc_reg(r, addr)) {

        unsigned index = ROCKER_RING_INDEX(addr);

        unsigned offset = addr & ROCKER_DMA_DESC_MASK;



        switch (offset) {

        case ROCKER_DMA_DESC_ADDR_OFFSET:

            r->lower32 = (uint64_t)val;

            break;

        case ROCKER_DMA_DESC_ADDR_OFFSET + 4:

            desc_ring_set_base_addr(r->rings[index],

                                    ((uint64_t)val) << 32 | r->lower32);

            r->lower32 = 0;

            break;

        case ROCKER_DMA_DESC_SIZE_OFFSET:

            desc_ring_set_size(r->rings[index], val);

            break;

        case ROCKER_DMA_DESC_HEAD_OFFSET:

            if (desc_ring_set_head(r->rings[index], val)) {

                rocker_msix_irq(r, desc_ring_get_msix_vector(r->rings[index]));

            }

            break;

        case ROCKER_DMA_DESC_CTRL_OFFSET:

            desc_ring_set_ctrl(r->rings[index], val);

            break;

        case ROCKER_DMA_DESC_CREDITS_OFFSET:

            if (desc_ring_ret_credits(r->rings[index], val)) {

                rocker_msix_irq(r, desc_ring_get_msix_vector(r->rings[index]));

            }

            break;

        default:

            DPRINTF("not implemented dma reg write(l) addr=0x" TARGET_FMT_plx

                    " val=0x%08x (ring %d, addr=0x%02x)\n",

                    addr, val, index, offset);

            break;

        }

        return;

    }



    switch (addr) {

    case ROCKER_TEST_REG:

        r->test_reg = val;

        break;

    case ROCKER_TEST_REG64:

    case ROCKER_TEST_DMA_ADDR:

    case ROCKER_PORT_PHYS_ENABLE:

        r->lower32 = (uint64_t)val;

        break;

    case ROCKER_TEST_REG64 + 4:

        r->test_reg64 = ((uint64_t)val) << 32 | r->lower32;

        r->lower32 = 0;

        break;

    case ROCKER_TEST_IRQ:

        rocker_msix_irq(r, val);

        break;

    case ROCKER_TEST_DMA_SIZE:

        r->test_dma_size = val;

        break;

    case ROCKER_TEST_DMA_ADDR + 4:

        r->test_dma_addr = ((uint64_t)val) << 32 | r->lower32;

        r->lower32 = 0;

        break;

    case ROCKER_TEST_DMA_CTRL:

        rocker_test_dma_ctrl(r, val);

        break;

    case ROCKER_CONTROL:

        rocker_control(r, val);

        break;

    case ROCKER_PORT_PHYS_ENABLE + 4:

        rocker_port_phys_enable_write(r, ((uint64_t)val) << 32 | r->lower32);

        r->lower32 = 0;

        break;

    default:

        DPRINTF("not implemented write(l) addr=0x" TARGET_FMT_plx

                " val=0x%08x\n", addr, val);

        break;

    }

}