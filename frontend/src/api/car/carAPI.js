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
};

export default carAPI;
