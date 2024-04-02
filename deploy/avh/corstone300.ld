/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*------------------ Reference System Memories -------------
  +===================+============+=======+============+============+
  | Memory            | Address    | Size  | CPU Access | NPU Access |
  +===================+============+=======+============+============+
  | ITCM              | 0x00000000 | 512KB | Yes (RO)   | No         |
  +-------------------+------------+-------+------------+------------+
  | DTCM              | 0x20000000 | 512KB | Yes (R/W)  | No         |
  +-------------------+------------+-------+------------+------------+
  | SSE-300 SRAM      | 0x21000000 |   2MB | Yes (R/W)  | Yes (R/W)  |
  +-------------------+------------+-------+------------+------------+
  | Data SRAM         | 0x01000000 |   2MB | Yes (R/W)  | Yes (R/W)  |
  +-------------------+------------+-------+------------+------------+
  | DDR               | 0x60000000 |  32MB | Yes (R/W)  | Yes (R/W)  |
  +-------------------+------------+-------+------------+------------+ */

/*---------------------- ITCM Configuration ----------------------------------
  <h> Flash Configuration
    <o0> Flash Base Address <0x0-0xFFFFFFFF:8>
    <o1> Flash Size (in Bytes) <0x0-0xFFFFFFFF:8>
  </h>
  -----------------------------------------------------------------------------*/
__ROM_BASE = 0x00000000;
__ROM_SIZE = 0x00080000;

/*--------------------- DTCM RAM Configuration ----------------------------
  <h> RAM Configuration
    <o0> RAM Base Address    <0x0-0xFFFFFFFF:8>
    <o1> RAM Size (in Bytes) <0x0-0xFFFFFFFF:8>
  </h>
 -----------------------------------------------------------------------------*/
__RAM_BASE = 0x20000000;
__RAM_SIZE = 0x00080000;

/*----------------------- Data SRAM Configuration ------------------------------
  <h> Data SRAM Configuration
    <o0> DATA_SRAM Base Address    <0x0-0xFFFFFFFF:8>
    <o1> DATA_SRAM Size (in Bytes) <0x0-0xFFFFFFFF:8>
  </h>
 -----------------------------------------------------------------------------*/
__DATA_SRAM_BASE = 0x01000000;
__DATA_SRAM_SIZE = 0x00200000;

/*--------------------- Embedded SRAM Configuration ----------------------------
  <h> SRAM Configuration
    <o0> SRAM Base Address    <0x0-0xFFFFFFFF:8>
    <o1> SRAM Size (in Bytes) <0x0-0xFFFFFFFF:8>
  </h>
 -----------------------------------------------------------------------------*/
__SRAM_BASE = 0x21000000;
__SRAM_SIZE = 0x00200000;

/*--------------------- Stack / Heap Configuration ----------------------------
  <h> Stack / Heap Configuration
    <o0> Stack Size (in Bytes) <0x0-0xFFFFFFFF:8>
    <o1> Heap Size (in Bytes) <0x0-0xFFFFFFFF:8>
  </h>
  -----------------------------------------------------------------------------*/
__STACK_SIZE = 0x00008000;
__HEAP_SIZE  = 0x00008000;

/*--------------------- Embedded RAM Configuration ----------------------------
  <h> DDR Configuration
    <o0> DDR Base Address    <0x0-0xFFFFFFFF:8>
    <o1> DDR Size (in Bytes) <0x0-0xFFFFFFFF:8>
  </h>
 -----------------------------------------------------------------------------*/
__DDR_BASE = 0x60000000;
__DDR_SIZE = 0x02000000;

/*
 *-------------------- <<< end of configuration section >>> -------------------
 */

MEMORY
{
  ITCM       (rx)  : ORIGIN = __ROM_BASE, LENGTH = __ROM_SIZE
  DTCM       (rwx) : ORIGIN = __RAM_BASE, LENGTH = __RAM_SIZE
  DATA_SRAM  (rwx) : ORIGIN = __DATA_SRAM_BASE, LENGTH = __DATA_SRAM_SIZE
  SRAM       (rwx) : ORIGIN = __SRAM_BASE, LENGTH = __SRAM_SIZE
  DDR        (rwx) : ORIGIN = __DDR_BASE, LENGTH = __DDR_SIZE
}

/* Linker script to place sections and symbol values. Should be used together
 * with other linker script that defines memory regions ITCM and RAM.
 * It references following symbols, which must be defined in code:
 *   Reset_Handler : Entry of reset handler
 *
 * It defines following symbols, which code can use without definition:
 *   __exidx_start
 *   __exidx_end
 *   __copy_table_start__
 *   __copy_table_end__
 *   __zero_table_start__
 *   __zero_table_end__
 *   __etext
 *   __data_start__
 *   __preinit_array_start
 *   __preinit_array_end
 *   __init_array_start
 *   __init_array_end
 *   __fini_array_start
 *   __fini_array_end
 *   __data_end__
 *   __bss_start__
 *   __bss_end__
 *   __end__
 *   end
 *   __HeapLimit
 *   __StackLimit
 *   __StackTop
 *   __stack
 */
ENTRY(Reset_Handler)

SECTIONS
{
  /* .ddr is placed before .text so that .rodata.tvm is encountered before .rodata* */
  .ddr :
  {
    . = ALIGN (16);
    *(.rodata.tvm)
    . = ALIGN (16);
    *(.data.tvm);
    . = ALIGN(16);    
  } > DDR

  .text :
  {
    KEEP(*(.vectors))
    *(.text*)

    KEEP(*(.init))
    KEEP(*(.fini))

    /* .ctors */
    *crtbegin.o(.ctors)
    *crtbegin?.o(.ctors)
    *(EXCLUDE_FILE(*crtend?.o *crtend.o) .ctors)
    *(SORT(.ctors.*))
    *(.ctors)

    /* .dtors */
    *crtbegin.o(.dtors)
    *crtbegin?.o(.dtors)
    *(EXCLUDE_FILE(*crtend?.o *crtend.o) .dtors)
    *(SORT(.dtors.*))
    *(.dtors)

    *(.rodata*)

    KEEP(*(.eh_frame*))
  } > ITCM

  .ARM.extab :
  {
    *(.ARM.extab* .gnu.linkonce.armextab.*)
  } > ITCM

  __exidx_start = .;
  .ARM.exidx :
  {
    *(.ARM.exidx* .gnu.linkonce.armexidx.*)
  } > ITCM
  __exidx_end = .;

  .copy.table :
  {
    . = ALIGN(4);
    __copy_table_start__ = .;
    LONG (__etext)
    LONG (__data_start__)
    LONG (__data_end__ - __data_start__)
    /* Add each additional data section here */
    __copy_table_end__ = .;
  } > ITCM

  .zero.table :
  {
    . = ALIGN(4);
    __zero_table_start__ = .;
    __zero_table_end__ = .;
  } > ITCM

  /**
   * Location counter can end up 2byte aligned with narrow Thumb code but
   * __etext is assumed by startup code to be the LMA of a section in DTCM
   * which must be 4byte aligned
   */
  __etext = ALIGN (4);

  .sram :
  {
    . = ALIGN(16);
  } > SRAM AT > SRAM

  .data : AT (__etext)
  {
    __data_start__ = .;
    *(vtable)
    *(.data)
    *(.data.*)

    . = ALIGN(4);
    /* preinit data */
    PROVIDE_HIDDEN (__preinit_array_start = .);
    KEEP(*(.preinit_array))
    PROVIDE_HIDDEN (__preinit_array_end = .);

    . = ALIGN(4);
    /* init data */
    PROVIDE_HIDDEN (__init_array_start = .);
    KEEP(*(SORT(.init_array.*)))
    KEEP(*(.init_array))
    PROVIDE_HIDDEN (__init_array_end = .);


    . = ALIGN(4);
    /* finit data */
    PROVIDE_HIDDEN (__fini_array_start = .);
    KEEP(*(SORT(.fini_array.*)))
    KEEP(*(.fini_array))
    PROVIDE_HIDDEN (__fini_array_end = .);

    KEEP(*(.jcr*))
    . = ALIGN(4);
    /* All data end */
    __data_end__ = .;

  } > DTCM

  .bss.noinit (NOLOAD):
  {
    . = ALIGN(16);
    *(.bss.noinit.*)
    . = ALIGN(16);
  } > SRAM AT > SRAM

  .bss :
  {
    . = ALIGN(4);
    __bss_start__ = .;
    *(.bss)
    *(.bss.*)
    *(COMMON)
    . = ALIGN(4);
    __bss_end__ = .;
  } > DTCM AT > DTCM

  .data_sram :
  {
    . = ALIGN(16);
  } > DATA_SRAM

  .heap (COPY) :
  {
    . = ALIGN(8);
    __end__ = .;
    PROVIDE(end = .);
    . = . + __HEAP_SIZE;
    . = ALIGN(8);
    __HeapLimit = .;
  } > DTCM

  .stack (ORIGIN(DTCM) + LENGTH(DTCM) - __STACK_SIZE) (COPY) :
  {
    . = ALIGN(8);
    __StackLimit = .;
    . = . + __STACK_SIZE;
    . = ALIGN(8);
    __StackTop = .;
  } > DTCM
  PROVIDE(__stack = __StackTop);

  /* Check if data + stack exceeds DTCM limit */
  ASSERT(__StackLimit >= __bss_end__, "region DTCM overflowed with stack")
}
