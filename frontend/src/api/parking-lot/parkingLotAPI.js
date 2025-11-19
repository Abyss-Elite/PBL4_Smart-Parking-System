import api from "@/lib/axiosInstance";

export const parkingLotAPI = {
  getCurrentVehicleCondition: () => {
    return api.get("/parkingLot/summary");
  },
  getUsageInfo: () => {
    return api.get("/parkingLot/usageInfo");
  },
  getInitialStartDate: () => {
    return api.get("/parkingLot/firstActiveDate");
  },
};
