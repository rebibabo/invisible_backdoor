static void ohci_bus_stop(OHCIState *ohci)

{

    trace_usb_ohci_stop(ohci->name);

    if (ohci->eof_timer) {

        timer_del(ohci->eof_timer);

        timer_free(ohci->eof_timer);

    }

    ohci->eof_timer = NULL;

}
