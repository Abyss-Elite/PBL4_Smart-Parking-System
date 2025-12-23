import api from "@/lib/axiosInstance";

export const userBookingPageAPI = {
  vnpayPayment: ({ amount, bookingId }) => {
    return api.post(
      `/vnpay/create?bookingId=${encodeURIComponent(bookingId)}&amount=${encodeURIComponent(amount)}`
    );
  },
  paymentReturn: (data) => {
    const formData = new URLSearchParams();
    for (const key in data) {
      formData.append(key, data[key]);
    }
    return api.post("vnpay/return", formData, {
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
    });
  },
};
