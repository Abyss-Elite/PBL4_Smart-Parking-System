import api from "@/lib/axiosInstance";

export const bookingAPI = {
  createBooking: (booking) => {
    return api.post("/bookings",booking);
  },
  getBooking: (bookingId) => {
    return api.get(`/bookings/${bookingId}`);
  }
}