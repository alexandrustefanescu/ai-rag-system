import { auth } from "@/auth";

export default auth((req) => {
    if (!req.auth) {
        return Response.json({ error: "Unauthorized" }, { status: 401 });
    }
});

export const config = {
    matcher: ["/((?!api|_next/static|_next/image|favicon.ico|login|api/auth).*)"],
};
