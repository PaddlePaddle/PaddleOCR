function writeString(target, offset, length, value) {
  const encoded = new TextEncoder().encode(value);
  target.set(encoded.slice(0, length), offset);
}

function writeOctal(target, offset, length, value) {
  const octal = value.toString(8).padStart(length - 1, "0");
  writeString(target, offset, length, `${octal}\0`);
}

export function createTar(entries) {
  const chunks = [];
  for (const entry of entries) {
    const data =
      typeof entry.content === "string" ? new TextEncoder().encode(entry.content) : entry.content;
    const header = new Uint8Array(512);
    writeString(header, 0, 100, entry.name);
    writeString(header, 100, 8, "0000777");
    writeString(header, 108, 8, "0000000");
    writeString(header, 116, 8, "0000000");
    writeOctal(header, 124, 12, data.length);
    writeString(header, 136, 12, "00000000000");
    header[156] = "0".charCodeAt(0);
    writeString(header, 257, 6, "ustar");
    writeString(header, 263, 2, "00");
    for (let index = 148; index < 156; index += 1) {
      header[index] = 32;
    }
    const checksum = header.reduce((sum, value) => sum + value, 0);
    writeString(header, 148, 8, `${checksum.toString(8).padStart(6, "0")}\0 `);
    chunks.push(header, data);
    const remainder = data.length % 512;
    if (remainder !== 0) {
      chunks.push(new Uint8Array(512 - remainder));
    }
  }
  chunks.push(new Uint8Array(1024));

  const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out.buffer;
}
