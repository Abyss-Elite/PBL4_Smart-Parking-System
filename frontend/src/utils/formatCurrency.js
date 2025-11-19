export function formatCurrency(amount, locale = "vi-VN", currency = "VND") {
  if (isNaN(amount)) return "0 ₫";
  return amount.toLocaleString(locale, {
    style: "currency",
    currency,
  });
}
