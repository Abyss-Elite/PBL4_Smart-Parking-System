import api from "@/lib/axiosInstance";

export const carAPI = {
  getCarByUserId: (id) => {
    return api.get(`/car/byUser/${id}`);
  },
  getCurrentMonitoredCar: () => {
    return api.get("/parkingLot/currentPlate");
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
};

