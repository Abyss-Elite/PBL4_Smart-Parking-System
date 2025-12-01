const PATH = {
  LOGIN: "/login",
  REGISTER: "/register",
  DASHBOARD: {
    ADMIN_HOME: "/dashboard/admin_home",
    ON_DUTY_STAFF_MANAGEMENT: "/dashboard/on_duty_staff_management",
    TRANSACTION_HISTORY: "/dashboard/transaction_history",
    REPORTS_AND_ANALYTICS: "/dashboard/reports_and_analytics",
    PARKING_LOT_MANAGEMENT: {
      HOME: "/dashboard/parking_lot_management",
      Gate_Surveillance: "/dashboard/parking_lot_management/Gate_surveillance",
      RESERVATION: {
        HOME: "/dashboard/parking_lot_management/reservation",
        DETAIL: (id) => `/dashboard/parking_lot_management/reservation/${id}`,
      },
    },
    SCHEDULE: "/dashboard/schedule",
    SETTING: "/dashboard/setting",
    PROFILE: "/dashboard/profile",
  },
  PARKING_RESERVATION: {
    PAYMENT: "/parkingReservation/payment",
    PENDINGBILL: (id) => `/parkingReservation/pending-bill/${id}`,
  }
};

export default PATH;
