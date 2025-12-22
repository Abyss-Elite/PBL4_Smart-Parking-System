import axios from "axios";

export const userBookingPageAPI = {
  vnpayPayment: ({ amount, orderId }) => {
    return axios.post(
      `http://192.168.1.5:8084/api/v1/vnpay/create?amount=${encodeURIComponent(amount)}&orderId=${encodeURIComponent(orderId)}`
    );
  },
};
