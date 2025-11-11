import api from "@/lib/axiosInstance";

const carAPI = {
  getCarByUserId: (id) => {
    return api.get(`/car/byUser/${id}`);
  },
  getUsageInfo: () => {
    return api.get("/parkingLot/usageInfo");
  },
  getCurrentMonitoredCar: () => {
    return api.get("/parkingLot/currentPlate");
  },
  getReservatedSpot: (id, startOfWeek) => {
    return api.get(`/parkingSpot/${id}/bookings`, {
      params: {startOfWeek},
    });
  },
  getBookingDetail: (bookingId)=>{
    return api.get(`/parkingLot/${bookingId}`)
  },
  getAllSpotReservation: ()=>{
    return api.get("/parkingLot/2/spots");
  }
};

export default carAPI;
