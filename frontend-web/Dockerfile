# Use official Node.js lightweight image
FROM node:18-alpine

# Build-time env: Next.js inlines NEXT_PUBLIC_* at `next build`.
ARG NEXT_PUBLIC_RESEARCH_API_URL=http://127.0.0.1:8001
ENV NEXT_PUBLIC_RESEARCH_API_URL=$NEXT_PUBLIC_RESEARCH_API_URL

# Set working directory
WORKDIR /app

# Copy package files and install dependencies
COPY package.json package-lock.json* ./
RUN npm install

# Copy all frontend source files
COPY . .

# Build the Next.js application
RUN npm run build

# Expose the port Next.js runs on
EXPOSE 3000

# Start the application
CMD ["npm", "run", "start"]