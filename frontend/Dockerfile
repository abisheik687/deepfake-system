# Internal trace:
# - Wrong before: the frontend container ran the dev server directly and did not match the cleaned production-oriented entry path.
# - Fixed now: the image builds the Vite app and serves it through `vite preview` on the expected port.

FROM node:20

WORKDIR /app

COPY package*.json /app/
RUN npm install

COPY . /app

RUN npm run build

EXPOSE 4173

CMD ["npm", "run", "preview", "--", "--host", "0.0.0.0", "--port", "4173"]
