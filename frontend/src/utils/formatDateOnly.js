export function formatDateOnly(dateInput) {
  if (!dateInput) return null;

  // Chuyển chuỗi hoặc Date thành đối tượng Date
  const dateObj = new Date(dateInput);

  if (isNaN(dateObj)) return null; // Nếu không phải ngày hợp lệ, trả về null

  // Trích xuất năm-tháng-ngày
  const year = dateObj.getFullYear();
  const month = String(dateObj.getMonth() + 1).padStart(2, "0"); // Tháng từ 0–11
  const day = String(dateObj.getDate()).padStart(2, "0");

  return `${year}-${month}-${day}`;
}
