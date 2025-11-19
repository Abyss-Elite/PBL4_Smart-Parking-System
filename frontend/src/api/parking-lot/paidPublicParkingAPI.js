import api from "@/lib/axiosInstance";

export const paidPublicParkingAPI = {
  getParkingCondition: () => {
    return api.get("/parkingLot/1/summaryToday");
  },
  searchPlate: (licensePlate) => {
    return api.get("/parkingLot/1/search", {
      params: {
        licensePlate: licensePlate,
      },
    });
  },
  getCarsInLot: () => {
    return api.get("/parkingLot/1/carsInLot");
  },
};
