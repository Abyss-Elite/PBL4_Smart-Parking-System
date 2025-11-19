export function percentageToInt(percent) {
  if (!percent) return 0;
  const number = parseFloat(percent.toString().replace("%", "").trim());
  return Math.round(number);
}
