import api from "@/lib/axiosInstance";

export const bookingAPI = {
  createBooking: (booking) => {
    return api.post("booking",booking);
  },
  getBooking: (bookingId) => {
    return api.get(`booking/${bookingId}`);
  }
}