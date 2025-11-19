export function validatePlate(plate) {
  const regex = /^[0-9]{2}[A-Z]{1,2}[0-9]{4,5}$/i;
  return regex.test(plate.replace(/[-\s]/g, ""));
}
