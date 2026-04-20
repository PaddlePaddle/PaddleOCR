/*
 * Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

function readString(bytes: Uint8Array, start: number, length: number): string {
  let output = "";
  for (let index = start; index < start + length; index += 1) {
    const value = bytes[index];
    if (value === 0) break;
    output += String.fromCharCode(value);
  }
  return output.replace(/\0.*$/, "").trim();
}

function readOctal(bytes: Uint8Array, start: number, length: number): number {
  const raw = readString(bytes, start, length).replace(/\0/g, "").trim();
  return raw ? Number.parseInt(raw, 8) : 0;
}

function isEmptyBlock(bytes: Uint8Array, offset: number): boolean {
  for (let index = offset; index < offset + 512; index += 1) {
    if (bytes[index] !== 0) return false;
  }
  return true;
}

function normalizeEntryName(name: string): string {
  return name.replace(/^\.?\//, "");
}

function isMetadataEntry(name: string): boolean {
  const segments = normalizeEntryName(name).split("/");
  const baseName = segments[segments.length - 1] || "";
  return (
    baseName.startsWith("._") || segments.includes("PaxHeader") || segments.includes("__MACOSX")
  );
}

export function extractTarEntries(buffer: ArrayBuffer | Uint8Array): Map<string, Uint8Array> {
  const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  const entries = new Map<string, Uint8Array>();
  let offset = 0;

  while (offset + 512 <= bytes.length) {
    if (isEmptyBlock(bytes, offset)) {
      break;
    }

    const name = normalizeEntryName(readString(bytes, offset, 100));
    const size = readOctal(bytes, offset + 124, 12);
    const type = bytes[offset + 156];
    const dataStart = offset + 512;
    const dataEnd = dataStart + size;

    if (type !== 53 && type !== 120 && name && !isMetadataEntry(name)) {
      entries.set(name, bytes.slice(dataStart, dataEnd));
    }

    offset = dataStart + Math.ceil(size / 512) * 512;
  }

  return entries;
}

export function pickTarEntry(entries: Map<string, Uint8Array>, targetName: string): Uint8Array {
  const normalizedTarget = normalizeEntryName(targetName);
  const entry = entries.get(normalizedTarget);
  if (entry) {
    return entry;
  }

  for (const [name, value] of entries) {
    if (name.endsWith(`/${normalizedTarget}`) || name === normalizedTarget) {
      return value;
    }
  }

  throw new Error(`Entry "${targetName}" was not found in the tar archive.`);
}
