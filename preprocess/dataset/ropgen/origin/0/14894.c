static void sdhci_sdma_transfer_multi_blocks(SDHCIState *s)

{

    bool page_aligned = false;

    unsigned int n, begin;

    const uint16_t block_size = s->blksize & 0x0fff;

    uint32_t boundary_chk = 1 << (((s->blksize & 0xf000) >> 12) + 12);

    uint32_t boundary_count = boundary_chk - (s->sdmasysad % boundary_chk);



    /* XXX: Some sd/mmc drivers (for example, u-boot-slp) do not account for

     * possible stop at page boundary if initial address is not page aligned,

     * allow them to work properly */

    if ((s->sdmasysad % boundary_chk) == 0) {

        page_aligned = true;

    }



    if (s->trnmod & SDHC_TRNS_READ) {

        s->prnsts |= SDHC_DOING_READ | SDHC_DATA_INHIBIT |

                SDHC_DAT_LINE_ACTIVE;

        while (s->blkcnt) {

            if (s->data_count == 0) {

                for (n = 0; n < block_size; n++) {

                    s->fifo_buffer[n] = sdbus_read_data(&s->sdbus);

                }

            }

            begin = s->data_count;

            if (((boundary_count + begin) < block_size) && page_aligned) {

                s->data_count = boundary_count + begin;

                boundary_count = 0;

             } else {

                s->data_count = block_size;

                boundary_count -= block_size - begin;

                if (s->trnmod & SDHC_TRNS_BLK_CNT_EN) {

                    s->blkcnt--;

                }

            }

            dma_memory_write(&address_space_memory, s->sdmasysad,

                             &s->fifo_buffer[begin], s->data_count - begin);

            s->sdmasysad += s->data_count - begin;

            if (s->data_count == block_size) {

                s->data_count = 0;

            }

            if (page_aligned && boundary_count == 0) {

                break;

            }

        }

    } else {

        s->prnsts |= SDHC_DOING_WRITE | SDHC_DATA_INHIBIT |

                SDHC_DAT_LINE_ACTIVE;

        while (s->blkcnt) {

            begin = s->data_count;

            if (((boundary_count + begin) < block_size) && page_aligned) {

                s->data_count = boundary_count + begin;

                boundary_count = 0;

             } else {

                s->data_count = block_size;

                boundary_count -= block_size - begin;

            }

            dma_memory_read(&address_space_memory, s->sdmasysad,

                            &s->fifo_buffer[begin], s->data_count);

            s->sdmasysad += s->data_count - begin;

            if (s->data_count == block_size) {

                for (n = 0; n < block_size; n++) {

                    sdbus_write_data(&s->sdbus, s->fifo_buffer[n]);

                }

                s->data_count = 0;

                if (s->trnmod & SDHC_TRNS_BLK_CNT_EN) {

                    s->blkcnt--;

                }

            }

            if (page_aligned && boundary_count == 0) {

                break;

            }

        }

    }



    if (s->blkcnt == 0) {

        sdhci_end_transfer(s);

    } else {

        if (s->norintstsen & SDHC_NISEN_DMA) {

            s->norintsts |= SDHC_NIS_DMA;

        }

        sdhci_update_irq(s);

    }

}
