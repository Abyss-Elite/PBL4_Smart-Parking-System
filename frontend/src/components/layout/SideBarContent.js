import Link from "next/link";
import { usePathname } from "next/navigation";
import { IconHome } from "../ui/icon/IconHome";
import { IconParkingLotManagement } from "../ui/icon/IconParkingLotManagement";
import { IconEnployee } from "../ui/icon/IconEmployee";
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
      icon: <IconEnployee />,
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

    const linkClass = `
      flex items-center gap-2 px-3 py-2 rounded-lg transition-colors duration-300
      ${
        isActive
          ? "bg-[#F3F4F6] dark:bg-gray-700 text-green-600 font-semibold"
          : "text-[#747778] hover:text-green-600 hover:font-semibold hover:bg-[#F3F4F6] dark:text-gray-300 dark:hover:text-gray-100 dark:hover:bg-gray-800"
      }
    `;

    return (
      <Link key={path} href={path} className={linkClass}>
        {icon}
        <span>{label}</span>
      </Link>
    );
  };

  return <div className="flex flex-col gap-2 px-4">{navItems.map(renderNavLink)}</div>;
}
