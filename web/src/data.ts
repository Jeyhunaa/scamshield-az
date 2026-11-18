import type { ActivityItem } from "./types";

// Fictional customer, bank and account data for the simulated app.
export const BANK_NAME = "Karvan";
export const CUSTOMER = { firstName: "Aysel", fullName: "Aysel Hasanova", phoneMasked: "+994 •• ••• •• 47" };
export const ACCOUNT = { name: "Current account", iban: "AZ12KRVN40060019440012374471", openingBalance: 12480.35 };
export const DASHBOARD_URL = (import.meta.env.VITE_DASHBOARD_URL as string | undefined) ?? "http://localhost:8501";

export const SAVED_RECIPIENTS = [
  { recipient: "Sevinj Karimova", iban: "AZ30KRVN40060019440055120871", bank: "Karvan" },
  { recipient: "Rashad Guliyev", iban: "AZ81CSPN38810000000071230045", bank: "Caspian Savings" },
];

export const RECENT_ACTIVITY: ActivityItem[] = [
  { id: "a1", title: "Grocery store", subtitle: "Today, 09:12 · Card", amount: -38.4, kind: "grocery" },
  { id: "a2", title: "Coffee house", subtitle: "Yesterday, 08:47 · Card", amount: -6.5, kind: "coffee" },
  { id: "a3", title: "Salary — Caspian Logistics LLC", subtitle: "15 Nov · Incoming transfer", amount: 2450.0, kind: "salary" },
  { id: "a4", title: "Electricity bill", subtitle: "14 Nov · Bill payment", amount: -54.2, kind: "bill" },
  { id: "a5", title: "Sevinj Karimova", subtitle: "12 Nov · Transfer", amount: -150.0, kind: "transfer" },
  { id: "a6", title: "Online store", subtitle: "10 Nov · Card", amount: -89.99, kind: "shopping" },
];
