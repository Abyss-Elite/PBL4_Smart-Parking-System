import api from "@/lib/axiosInstance";

export const revenueAPI = {
  revenueCurrentMonth: () => {
    return api.get("/revenue/thisMonth");
  },

  revenueWeek: (date) => {
    return api.get(`/revenue/week?date=${date}`);
  },

  revenueYear: (year) => {
    return api.get(`revenue/year?year=${year}`);
  },

  revenueMonth: (year) => {
    return api.get(`revenue/monthly?year=${year}`);
  },
};
