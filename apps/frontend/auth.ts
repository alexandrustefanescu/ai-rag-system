import { auth } from "@clerk/nextjs/server";

export const { auth, signIn, signOut } = auth;

export const handlers = {
    GET: auth().GET,
    POST: auth().POST,
};
