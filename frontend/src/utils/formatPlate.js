export const formatLicensePlate = (value) => {
  let v = value.toUpperCase().replace(/[^A-Z0-9-]/g, "");

  const match = v.match(/^(\d{2}[A-Z]-)(\d{5})$/);

  if (match) {
    return `${match[1]}${match[2].slice(0, 3)}.${match[2].slice(3)}`;
  }

  return v;
};
