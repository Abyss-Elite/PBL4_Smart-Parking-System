import api from "@/lib/axiosInstance";
import axios from "axios";
import { CONFIG } from "@/lib/config";

export const carAPI = {
  getCarByUserId: (id) => {
    return api.get(`/car/byUser/${id}`);
  },
  getCurrentMonitoredCar: () => {
    return api.get("/parkingLot/currentPlate/stream");
  },
  getRecentActivitiesCar: () => {
    return api.get("/car/recentActivities");
  },
  getTotalBookedCars: () => {
    return api.get("/bookings/totalBookedCars");
  },
  getNextReservedCars: () => {
    return api.get("/bookings/nextReservedCars");
  },
  getReservatedSpot: (id, startOfWeek) => {
    return api.get(`/parkingSpot/${id}/bookings`, {
      params: { startOfWeek },
    });
  },
  getBookingDetail: (bookingId) => {
    return api.get(`/parkingLot/${bookingId}`);
  },
  getAllSpotReservation: () => {
    return api.get("/parkingLot/2/spots");
  },
  getAllBooking: () => {
    return axios.get(`${CONFIG.API_BASE_URL}/bookings`);
  },
};
