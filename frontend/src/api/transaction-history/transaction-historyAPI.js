import api from "@/lib/axiosInstance";

export const transaction_historyAPI = {
  getTransactionHistoryOfReservedParking: (keyword) => {
    return api.get("/transaction/2", {
      params: { keyword } 
    });
  },
  getTransactionHistoryOfFreeParking: (keyword) => {
    return api.get("/transaction/1", {
      params: { keyword }
    });
  },
};
