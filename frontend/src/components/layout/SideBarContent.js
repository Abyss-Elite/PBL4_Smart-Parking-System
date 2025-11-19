import Link from "next/link";
import { usePathname } from "next/navigation";
import { IconHome } from "../ui/icon/IconHome";
import { IconAccountManagement } from "../ui/icon/IconAccountManagement";
import { IconVehicleManagement } from "../ui/icon/IconVehicleManagement";
import { IconParkingLotManagement } from "../ui/icon/IconParkingLotManagement";
import { IconSecurityManagement } from "../ui/icon/IconSecurityManagement";
import { IconReport } from "../ui/icon/IconReport";
import PATH from "@/routes/PATH";

export function SideBarContent() {
  const pathname = usePathname();
  const navItems = [
    {
      label: "Dashboard",
      path: PATH.DASHBOARD.ADMIN_HOME,
      icon: <IconHome />,
    },
    {
      label: "Parking Lot Management",
      path: PATH.DASHBOARD.PARKING_LOT_MANAGEMENT.HOME,
      icon: <IconParkingLotManagement />,
    },
    {
      label: "On_Duty Staff Management",
      path: PATH.DASHBOARD.ON_DUTY_STAFF_MANAGEMENT,
      icon: <IconParkingLotManagement />,
    },
    {
      label: "Entry & Exit History",
      path: PATH.DASHBOARD.ENTRY_AND_EXIT_HISTORY,
      icon: <IconSecurityManagement />,
    },
    {
      label: "Reports & Analytics",
      path: PATH.DASHBOARD.REPORTS_AND_ANALYTICS,
      icon: <IconReport />,
    },
  ];

  const renderNavLink = ({ label, path, icon }) => {
    const isActive = pathname.startsWith(path);
    const itemClass = isActive
      ? "px-2 py-2 bg-[#F3F4F6] rounded-[6]"
      : "px-2 py-2 hover:bg-[#F3F4F6] rounded-[6]";
    const divClass = isActive
      ? "gap-2 flex items-center text-green-600 font-semibold"
      : "gap-2 flex items-center text-[#747778] hover:text-green-600 hover:font-semibold transition-all";

    return (
      <Link key={path} href={path} className={itemClass}>
        <div className={divClass}>
          {icon}
          <span>{label}</span>
        </div>
      </Link>
    );
  };

  return <div className="flex flex-col gap-2 px-4">{navItems.map(renderNavLink)}</div>;
}
