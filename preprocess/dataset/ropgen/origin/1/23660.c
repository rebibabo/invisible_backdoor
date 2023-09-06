static ssize_t rtl8139_do_receive(NetClientState *nc, const uint8_t *buf, size_t size_, int do_interrupt)

{

    RTL8139State *s = qemu_get_nic_opaque(nc);

    PCIDevice *d = PCI_DEVICE(s);

    /* size is the length of the buffer passed to the driver */

    int size = size_;

    const uint8_t *dot1q_buf = NULL;



    uint32_t packet_header = 0;



    uint8_t buf1[MIN_BUF_SIZE + VLAN_HLEN];

    static const uint8_t broadcast_macaddr[6] =

        { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff };



    DPRINTF(">>> received len=%d\n", size);



    /* test if board clock is stopped */

    if (!s->clock_enabled)

    {

        DPRINTF("stopped ==========================\n");

        return -1;

    }



    /* first check if receiver is enabled */



    if (!rtl8139_receiver_enabled(s))

    {

        DPRINTF("receiver disabled ================\n");

        return -1;

    }



    /* XXX: check this */

    if (s->RxConfig & AcceptAllPhys) {

        /* promiscuous: receive all */

        DPRINTF(">>> packet received in promiscuous mode\n");



    } else {

        if (!memcmp(buf,  broadcast_macaddr, 6)) {

            /* broadcast address */

            if (!(s->RxConfig & AcceptBroadcast))

            {

                DPRINTF(">>> broadcast packet rejected\n");



                /* update tally counter */

                ++s->tally_counters.RxERR;



                return size;

            }



            packet_header |= RxBroadcast;



            DPRINTF(">>> broadcast packet received\n");



            /* update tally counter */

            ++s->tally_counters.RxOkBrd;



        } else if (buf[0] & 0x01) {

            /* multicast */

            if (!(s->RxConfig & AcceptMulticast))

            {

                DPRINTF(">>> multicast packet rejected\n");



                /* update tally counter */

                ++s->tally_counters.RxERR;



                return size;

            }



            int mcast_idx = compute_mcast_idx(buf);



            if (!(s->mult[mcast_idx >> 3] & (1 << (mcast_idx & 7))))

            {

                DPRINTF(">>> multicast address mismatch\n");



                /* update tally counter */

                ++s->tally_counters.RxERR;



                return size;

            }



            packet_header |= RxMulticast;



            DPRINTF(">>> multicast packet received\n");



            /* update tally counter */

            ++s->tally_counters.RxOkMul;



        } else if (s->phys[0] == buf[0] &&

                   s->phys[1] == buf[1] &&

                   s->phys[2] == buf[2] &&

                   s->phys[3] == buf[3] &&

                   s->phys[4] == buf[4] &&

                   s->phys[5] == buf[5]) {

            /* match */

            if (!(s->RxConfig & AcceptMyPhys))

            {

                DPRINTF(">>> rejecting physical address matching packet\n");



                /* update tally counter */

                ++s->tally_counters.RxERR;



                return size;

            }



            packet_header |= RxPhysical;



            DPRINTF(">>> physical address matching packet received\n");



            /* update tally counter */

            ++s->tally_counters.RxOkPhy;



        } else {



            DPRINTF(">>> unknown packet\n");



            /* update tally counter */

            ++s->tally_counters.RxERR;



            return size;

        }

    }



    /* if too small buffer, then expand it

     * Include some tailroom in case a vlan tag is later removed. */

    if (size < MIN_BUF_SIZE + VLAN_HLEN) {

        memcpy(buf1, buf, size);

        memset(buf1 + size, 0, MIN_BUF_SIZE + VLAN_HLEN - size);

        buf = buf1;

        if (size < MIN_BUF_SIZE) {

            size = MIN_BUF_SIZE;

        }

    }



    if (rtl8139_cp_receiver_enabled(s))

    {

        if (!rtl8139_cp_rx_valid(s)) {

            return size;

        }



        DPRINTF("in C+ Rx mode ================\n");



        /* begin C+ receiver mode */



/* w0 ownership flag */

#define CP_RX_OWN (1<<31)

/* w0 end of ring flag */

#define CP_RX_EOR (1<<30)

/* w0 bits 0...12 : buffer size */

#define CP_RX_BUFFER_SIZE_MASK ((1<<13) - 1)

/* w1 tag available flag */

#define CP_RX_TAVA (1<<16)

/* w1 bits 0...15 : VLAN tag */

#define CP_RX_VLAN_TAG_MASK ((1<<16) - 1)

/* w2 low  32bit of Rx buffer ptr */

/* w3 high 32bit of Rx buffer ptr */



        int descriptor = s->currCPlusRxDesc;

        dma_addr_t cplus_rx_ring_desc;



        cplus_rx_ring_desc = rtl8139_addr64(s->RxRingAddrLO, s->RxRingAddrHI);

        cplus_rx_ring_desc += 16 * descriptor;



        DPRINTF("+++ C+ mode reading RX descriptor %d from host memory at "

            "%08x %08x = "DMA_ADDR_FMT"\n", descriptor, s->RxRingAddrHI,

            s->RxRingAddrLO, cplus_rx_ring_desc);



        uint32_t val, rxdw0,rxdw1,rxbufLO,rxbufHI;



        pci_dma_read(d, cplus_rx_ring_desc, &val, 4);

        rxdw0 = le32_to_cpu(val);

        pci_dma_read(d, cplus_rx_ring_desc+4, &val, 4);

        rxdw1 = le32_to_cpu(val);

        pci_dma_read(d, cplus_rx_ring_desc+8, &val, 4);

        rxbufLO = le32_to_cpu(val);

        pci_dma_read(d, cplus_rx_ring_desc+12, &val, 4);

        rxbufHI = le32_to_cpu(val);



        DPRINTF("+++ C+ mode RX descriptor %d %08x %08x %08x %08x\n",

            descriptor, rxdw0, rxdw1, rxbufLO, rxbufHI);



        if (!(rxdw0 & CP_RX_OWN))

        {

            DPRINTF("C+ Rx mode : descriptor %d is owned by host\n",

                descriptor);



            s->IntrStatus |= RxOverflow;

            ++s->RxMissed;



            /* update tally counter */

            ++s->tally_counters.RxERR;

            ++s->tally_counters.MissPkt;



            rtl8139_update_irq(s);

            return size_;

        }



        uint32_t rx_space = rxdw0 & CP_RX_BUFFER_SIZE_MASK;



        /* write VLAN info to descriptor variables. */

        if (s->CpCmd & CPlusRxVLAN && be16_to_cpup((uint16_t *)

                &buf[ETH_ALEN * 2]) == ETH_P_VLAN) {

            dot1q_buf = &buf[ETH_ALEN * 2];

            size -= VLAN_HLEN;

            /* if too small buffer, use the tailroom added duing expansion */

            if (size < MIN_BUF_SIZE) {

                size = MIN_BUF_SIZE;

            }



            rxdw1 &= ~CP_RX_VLAN_TAG_MASK;

            /* BE + ~le_to_cpu()~ + cpu_to_le() = BE */

            rxdw1 |= CP_RX_TAVA | le16_to_cpup((uint16_t *)

                &dot1q_buf[ETHER_TYPE_LEN]);



            DPRINTF("C+ Rx mode : extracted vlan tag with tci: ""%u\n",

                be16_to_cpup((uint16_t *)&dot1q_buf[ETHER_TYPE_LEN]));

        } else {

            /* reset VLAN tag flag */

            rxdw1 &= ~CP_RX_TAVA;

        }



        /* TODO: scatter the packet over available receive ring descriptors space */



        if (size+4 > rx_space)

        {

            DPRINTF("C+ Rx mode : descriptor %d size %d received %d + 4\n",

                descriptor, rx_space, size);



            s->IntrStatus |= RxOverflow;

            ++s->RxMissed;



            /* update tally counter */

            ++s->tally_counters.RxERR;

            ++s->tally_counters.MissPkt;



            rtl8139_update_irq(s);

            return size_;

        }



        dma_addr_t rx_addr = rtl8139_addr64(rxbufLO, rxbufHI);



        /* receive/copy to target memory */

        if (dot1q_buf) {

            pci_dma_write(d, rx_addr, buf, 2 * ETH_ALEN);

            pci_dma_write(d, rx_addr + 2 * ETH_ALEN,

                          buf + 2 * ETH_ALEN + VLAN_HLEN,

                          size - 2 * ETH_ALEN);

        } else {

            pci_dma_write(d, rx_addr, buf, size);

        }



        if (s->CpCmd & CPlusRxChkSum)

        {

            /* do some packet checksumming */

        }



        /* write checksum */

        val = cpu_to_le32(crc32(0, buf, size_));

        pci_dma_write(d, rx_addr+size, (uint8_t *)&val, 4);



/* first segment of received packet flag */

#define CP_RX_STATUS_FS (1<<29)

/* last segment of received packet flag */

#define CP_RX_STATUS_LS (1<<28)

/* multicast packet flag */

#define CP_RX_STATUS_MAR (1<<26)

/* physical-matching packet flag */

#define CP_RX_STATUS_PAM (1<<25)

/* broadcast packet flag */

#define CP_RX_STATUS_BAR (1<<24)

/* runt packet flag */

#define CP_RX_STATUS_RUNT (1<<19)

/* crc error flag */

#define CP_RX_STATUS_CRC (1<<18)

/* IP checksum error flag */

#define CP_RX_STATUS_IPF (1<<15)

/* UDP checksum error flag */

#define CP_RX_STATUS_UDPF (1<<14)

/* TCP checksum error flag */

#define CP_RX_STATUS_TCPF (1<<13)



        /* transfer ownership to target */

        rxdw0 &= ~CP_RX_OWN;



        /* set first segment bit */

        rxdw0 |= CP_RX_STATUS_FS;



        /* set last segment bit */

        rxdw0 |= CP_RX_STATUS_LS;



        /* set received packet type flags */

        if (packet_header & RxBroadcast)

            rxdw0 |= CP_RX_STATUS_BAR;

        if (packet_header & RxMulticast)

            rxdw0 |= CP_RX_STATUS_MAR;

        if (packet_header & RxPhysical)

            rxdw0 |= CP_RX_STATUS_PAM;



        /* set received size */

        rxdw0 &= ~CP_RX_BUFFER_SIZE_MASK;

        rxdw0 |= (size+4);



        /* update ring data */

        val = cpu_to_le32(rxdw0);

        pci_dma_write(d, cplus_rx_ring_desc, (uint8_t *)&val, 4);

        val = cpu_to_le32(rxdw1);

        pci_dma_write(d, cplus_rx_ring_desc+4, (uint8_t *)&val, 4);



        /* update tally counter */

        ++s->tally_counters.RxOk;



        /* seek to next Rx descriptor */

        if (rxdw0 & CP_RX_EOR)

        {

            s->currCPlusRxDesc = 0;

        }

        else

        {

            ++s->currCPlusRxDesc;

        }



        DPRINTF("done C+ Rx mode ----------------\n");



    }

    else

    {

        DPRINTF("in ring Rx mode ================\n");



        /* begin ring receiver mode */

        int avail = MOD2(s->RxBufferSize + s->RxBufPtr - s->RxBufAddr, s->RxBufferSize);



        /* if receiver buffer is empty then avail == 0 */



#define RX_ALIGN(x) (((x) + 3) & ~0x3)



        if (avail != 0 && RX_ALIGN(size + 8) >= avail)

        {

            DPRINTF("rx overflow: rx buffer length %d head 0x%04x "

                "read 0x%04x === available 0x%04x need 0x%04x\n",

                s->RxBufferSize, s->RxBufAddr, s->RxBufPtr, avail, size + 8);



            s->IntrStatus |= RxOverflow;

            ++s->RxMissed;

            rtl8139_update_irq(s);

            return size_;

        }



        packet_header |= RxStatusOK;



        packet_header |= (((size+4) << 16) & 0xffff0000);



        /* write header */

        uint32_t val = cpu_to_le32(packet_header);



        rtl8139_write_buffer(s, (uint8_t *)&val, 4);



        rtl8139_write_buffer(s, buf, size);



        /* write checksum */

        val = cpu_to_le32(crc32(0, buf, size));

        rtl8139_write_buffer(s, (uint8_t *)&val, 4);



        /* correct buffer write pointer */

        s->RxBufAddr = MOD2(RX_ALIGN(s->RxBufAddr), s->RxBufferSize);



        /* now we can signal we have received something */



        DPRINTF("received: rx buffer length %d head 0x%04x read 0x%04x\n",

            s->RxBufferSize, s->RxBufAddr, s->RxBufPtr);

    }



    s->IntrStatus |= RxOK;



    if (do_interrupt)

    {

        rtl8139_update_irq(s);

    }



    return size_;

}
