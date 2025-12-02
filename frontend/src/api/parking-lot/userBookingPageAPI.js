import axios from "axios";

export const userBookingPageAPI = {
  vnpayPayment: ({ amount, orderId }) => {
    return axios.post(
      `/v1/vnpay/create?amount=${encodeURIComponent(amount)}&orderId=${encodeURIComponent(orderId)}`
    );
  },
};
