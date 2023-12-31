void *qemu_ram_mmap(int fd, size_t size, size_t align, bool shared)

{

    /*

     * Note: this always allocates at least one extra page of virtual address

     * space, even if size is already aligned.

     */

    size_t total = size + align;

    void *ptr = mmap(0, total, PROT_NONE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);

    size_t offset = QEMU_ALIGN_UP((uintptr_t)ptr, align) - (uintptr_t)ptr;

    void *ptr1;



    if (ptr == MAP_FAILED) {

        return NULL;

    }



    /* Make sure align is a power of 2 */

    assert(!(align & (align - 1)));

    /* Always align to host page size */

    assert(align >= getpagesize());



    ptr1 = mmap(ptr + offset, size, PROT_READ | PROT_WRITE,

                MAP_FIXED |

                (fd == -1 ? MAP_ANONYMOUS : 0) |

                (shared ? MAP_SHARED : MAP_PRIVATE),

                fd, 0);

    if (ptr1 == MAP_FAILED) {

        munmap(ptr, total);

        return NULL;

    }



    ptr += offset;

    total -= offset;



    if (offset > 0) {

        munmap(ptr - offset, offset);

    }



    /*

     * Leave a single PROT_NONE page allocated after the RAM block, to serve as

     * a guard page guarding against potential buffer overflows.

     */

    if (total > size + getpagesize()) {

        munmap(ptr + size + getpagesize(), total - size - getpagesize());

    }



    return ptr;

}
