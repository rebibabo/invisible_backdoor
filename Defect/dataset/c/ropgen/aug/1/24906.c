e1000_receive_iov(NetClientState *nc, const struct iovec *iov, int iovcnt)

{

    E1000State *s = qemu_get_nic_opaque(nc);

    PCIDevice *d = PCI_DEVICE(s);

    struct e1000_rx_desc desc;

    dma_addr_t base;

    unsigned int n, rdt;

    uint32_t rdh_start;

    uint16_t vlan_special = 0;

    uint8_t vlan_status = 0;

    uint8_t min_buf[MIN_BUF_SIZE];

    struct iovec min_iov;

    uint8_t *filter_buf = iov->iov_base;

    size_t size = iov_size(iov, iovcnt);

    size_t iov_ofs = 0;

    size_t desc_offset;

    size_t desc_size;

    size_t total_size;

    static const int PRCregs[6] = { PRC64, PRC127, PRC255, PRC511,

                                    PRC1023, PRC1522 };



    if (!(s->mac_reg[STATUS] & E1000_STATUS_LU)) {

        return -1;

    }



    if (!(s->mac_reg[RCTL] & E1000_RCTL_EN)) {

        return -1;

    }



    /* Pad to minimum Ethernet frame length */

    if (size < sizeof(min_buf)) {

        iov_to_buf(iov, iovcnt, 0, min_buf, size);

        memset(&min_buf[size], 0, sizeof(min_buf) - size);

        inc_reg_if_not_full(s, RUC);

        min_iov.iov_base = filter_buf = min_buf;

        min_iov.iov_len = size = sizeof(min_buf);

        iovcnt = 1;

        iov = &min_iov;

    } else if (iov->iov_len < MAXIMUM_ETHERNET_HDR_LEN) {

        /* This is very unlikely, but may happen. */

        iov_to_buf(iov, iovcnt, 0, min_buf, MAXIMUM_ETHERNET_HDR_LEN);

        filter_buf = min_buf;

    }



    /* Discard oversized packets if !LPE and !SBP. */

    if ((size > MAXIMUM_ETHERNET_LPE_SIZE ||

        (size > MAXIMUM_ETHERNET_VLAN_SIZE

        && !(s->mac_reg[RCTL] & E1000_RCTL_LPE)))

        && !(s->mac_reg[RCTL] & E1000_RCTL_SBP)) {

        inc_reg_if_not_full(s, ROC);

        return size;

    }



    if (!receive_filter(s, filter_buf, size)) {

        return size;

    }



    if (vlan_enabled(s) && is_vlan_packet(s, filter_buf)) {

        vlan_special = cpu_to_le16(be16_to_cpup((uint16_t *)(filter_buf

                                                                + 14)));

        iov_ofs = 4;

        if (filter_buf == iov->iov_base) {

            memmove(filter_buf + 4, filter_buf, 12);

        } else {

            iov_from_buf(iov, iovcnt, 4, filter_buf, 12);

            while (iov->iov_len <= iov_ofs) {

                iov_ofs -= iov->iov_len;

                iov++;

            }

        }

        vlan_status = E1000_RXD_STAT_VP;

        size -= 4;

    }



    rdh_start = s->mac_reg[RDH];

    desc_offset = 0;

    total_size = size + fcs_len(s);

    if (!e1000_has_rxbufs(s, total_size)) {

            set_ics(s, 0, E1000_ICS_RXO);

            return -1;

    }

    do {

        desc_size = total_size - desc_offset;

        if (desc_size > s->rxbuf_size) {

            desc_size = s->rxbuf_size;

        }

        base = rx_desc_base(s) + sizeof(desc) * s->mac_reg[RDH];

        pci_dma_read(d, base, &desc, sizeof(desc));

        desc.special = vlan_special;

        desc.status |= (vlan_status | E1000_RXD_STAT_DD);

        if (desc.buffer_addr) {

            if (desc_offset < size) {

                size_t iov_copy;

                hwaddr ba = le64_to_cpu(desc.buffer_addr);

                size_t copy_size = size - desc_offset;

                if (copy_size > s->rxbuf_size) {

                    copy_size = s->rxbuf_size;

                }

                do {

                    iov_copy = MIN(copy_size, iov->iov_len - iov_ofs);

                    pci_dma_write(d, ba, iov->iov_base + iov_ofs, iov_copy);

                    copy_size -= iov_copy;

                    ba += iov_copy;

                    iov_ofs += iov_copy;

                    if (iov_ofs == iov->iov_len) {

                        iov++;

                        iov_ofs = 0;

                    }

                } while (copy_size);

            }

            desc_offset += desc_size;

            desc.length = cpu_to_le16(desc_size);

            if (desc_offset >= total_size) {

                desc.status |= E1000_RXD_STAT_EOP | E1000_RXD_STAT_IXSM;

            } else {

                /* Guest zeroing out status is not a hardware requirement.

                   Clear EOP in case guest didn't do it. */

                desc.status &= ~E1000_RXD_STAT_EOP;

            }

        } else { // as per intel docs; skip descriptors with null buf addr

            DBGOUT(RX, "Null RX descriptor!!\n");

        }

        pci_dma_write(d, base, &desc, sizeof(desc));



        if (++s->mac_reg[RDH] * sizeof(desc) >= s->mac_reg[RDLEN])

            s->mac_reg[RDH] = 0;

        /* see comment in start_xmit; same here */

        if (s->mac_reg[RDH] == rdh_start) {

            DBGOUT(RXERR, "RDH wraparound @%x, RDT %x, RDLEN %x\n",

                   rdh_start, s->mac_reg[RDT], s->mac_reg[RDLEN]);

            set_ics(s, 0, E1000_ICS_RXO);

            return -1;

        }

    } while (desc_offset < total_size);



    increase_size_stats(s, PRCregs, total_size);

    inc_reg_if_not_full(s, TPR);

    s->mac_reg[GPRC] = s->mac_reg[TPR];

    /* TOR - Total Octets Received:

     * This register includes bytes received in a packet from the <Destination

     * Address> field through the <CRC> field, inclusively.

     * Always include FCS length (4) in size.

     */

    grow_8reg_if_not_full(s, TORL, size+4);

    s->mac_reg[GORCL] = s->mac_reg[TORL];

    s->mac_reg[GORCH] = s->mac_reg[TORH];



    n = E1000_ICS_RXT0;

    if ((rdt = s->mac_reg[RDT]) < s->mac_reg[RDH])

        rdt += s->mac_reg[RDLEN] / sizeof(desc);

    if (((rdt - s->mac_reg[RDH]) * sizeof(desc)) <= s->mac_reg[RDLEN] >>

        s->rxbuf_min_shift)

        n |= E1000_ICS_RXDMT0;



    set_ics(s, 0, n);



    return size;

}
