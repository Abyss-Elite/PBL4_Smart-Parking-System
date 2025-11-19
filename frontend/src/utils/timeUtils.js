export function isTimeOverlap(start1, end1, start2, end2) {
  const s1 = new Date(`2024-01-01 ${start1}`);
  const e1 = new Date(`2024-01-01 ${end1}`);
  const s2 = new Date(`2024-01-01 ${start2}`);
  const e2 = new Date(`2024-01-01 ${end2}`);

  return s1 < e2 && s2 < e1;
}
