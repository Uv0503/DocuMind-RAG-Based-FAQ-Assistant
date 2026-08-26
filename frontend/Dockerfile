# Production frontend image: install dependencies, compile Next.js, then serve the build.
FROM node:18-alpine

# Keeping application files under one directory makes later COPY and run steps predictable.
WORKDIR /app

# Copy dependency manifests first so Docker can reuse the install layer when source files change.
COPY package*.json ./
RUN npm install

# Application source is copied only after dependencies have been installed.
COPY . .

# Generate the optimized Next.js production bundle during image creation.
RUN npm run build

# Document the port used by the Next.js production server.
EXPOSE 3000

# Start the already-built application when the container launches.
CMD ["npm", "start"]
