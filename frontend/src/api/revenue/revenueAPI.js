import api from "@/lib/axiosInstance";

export const revenueAPI = {
  revenueCurrentMonth: () => {
    return api.get("/fees/total/current-month");
  },
};
