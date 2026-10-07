/** Largest image the model accepts. */
export const MAX_IMAGE_BYTES = 25 * 1024 * 1024;
const ACCEPTED_TYPES = ["image/png", "image/jpeg", "image/webp"];

export function dragHasImagePayload(dt: DataTransfer | null): boolean {
  if (!dt) return false;
  const types = Array.from(dt.types ?? []);
  return types.includes("Files") || types.includes("text/uri-list");
}

/** Throws a readable error if the file is not an image the model accepts. */
export function assertAcceptedImage(file: Blob): void {
  if (!ACCEPTED_TYPES.includes(file.type)) {
    throw new Error("Use a PNG, JPEG or WebP image.");
  }
  if (file.size === 0 || file.size > MAX_IMAGE_BYTES) {
    throw new Error("The image must be between 1 byte and 25 MiB.");
  }
}

/** Extracts an image from a drop: a file, or an image URL dragged from another page. */
export async function imageFileFromDataTransfer(dt: DataTransfer): Promise<File> {
  const file = Array.from(dt.files).find((f) => f.type.startsWith("image/"));
  if (file) return file;
  if (dt.files.length > 0) throw new Error("That file is not an image.");

  const uriList = dt.getData("text/uri-list");
  const uri =
    uriList
      .split("\n")
      .map((line) => line.trim())
      .find((line) => line && !line.startsWith("#")) ||
    dt.getData("text/plain").trim();

  if (!uri || !/^(https?:\/\/|data:image\/|\/)/.test(uri)) {
    throw new Error("Drop an image file or an image URL.");
  }

  let blob: Blob;
  try {
    const response = await fetch(uri);
    if (!response.ok) throw new Error();
    blob = await response.blob();
  } catch {
    throw new Error("Could not fetch that image. Save it locally and drop the file instead.");
  }

  if (!blob.type.startsWith("image/")) {
    throw new Error("That URL does not point to an image.");
  }

  const extension = blob.type.split("/")[1] || "jpg";
  return new File([blob], `dropped-image.${extension}`, { type: blob.type });
}
