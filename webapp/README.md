# Dynamic Investment Strategies Web Application

A comprehensive web application for Dynamic Investment Strategies with Market Regimes, built with FastAPI backend and Svelte frontend.

## Features

### Backend (FastAPI)
- **Authentication & Authorization**: JWT-based authentication with refresh tokens
- **Portfolio Management**: Create, optimize, and manage investment portfolios
- **Market Regime Analysis**: Identify and forecast market regimes using multiple methods
- **Backtesting Engine**: Comprehensive strategy backtesting with multi-horizon analysis
- **Scenario Analysis**: Monte Carlo simulations, stress testing, and historical replays
- **Market Data Integration**: Real-time and historical market data with FRED API
- **Analytics & Reporting**: Performance attribution, risk analysis, and custom reports
- **User Management**: User profiles, preferences, and activity tracking

### Frontend (Svelte)
- **Modern UI/UX**: Responsive design with Tailwind CSS
- **Interactive Dashboard**: Real-time portfolio metrics and market insights
- **Authentication Flow**: Login/register with form validation
- **Navigation**: Professional sidebar and header with notifications
- **Theme Support**: Light/dark mode switching
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## Technology Stack

### Backend
- **FastAPI**: Modern Python web framework with automatic API documentation
- **SQLAlchemy**: ORM for database operations
- **PostgreSQL**: Primary database for data persistence
- **Redis**: Caching and background task queue
- **Celery**: Distributed task queue for background processing
- **JWT**: JSON Web Tokens for authentication
- **Pydantic**: Data validation and settings management
- **Alembic**: Database migration tool

### Frontend
- **Svelte 4**: Modern reactive framework with TypeScript
- **Vite**: Fast build tool and development server
- **Tailwind CSS**: Utility-first CSS framework
- **Axios**: HTTP client with interceptors
- **Chart.js/D3.js**: Data visualization libraries
- **Lucide**: Beautiful icon library

### Infrastructure
- **Docker**: Containerization for all services
- **Docker Compose**: Multi-container application orchestration
- **Nginx**: Reverse proxy and static file serving
- **GitHub Actions**: CI/CD pipeline (ready for setup)

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Git
- Node.js 18+ (for local development)
- Python 3.11+ (for local development)

### Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Dynamic-Investment-Strategies-with-Market-Regimes/webapp
   ```

2. **Environment Configuration**
   ```bash
   # Backend environment
   cp backend/.env.example backend/.env
   
   # Frontend environment
   cp frontend/.env.example frontend/.env
   ```

3. **Start with Docker Compose**
   ```bash
   # Start all services
   docker-compose up -d
   
   # View logs
   docker-compose logs -f
   ```

4. **Access the Application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Database: localhost:5432 (postgres/regimeshift/regimeshift_password)

### Local Development

#### Backend Development
```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL="postgresql://regimeshift:regimeshift_password@localhost:5432/regimeshift"
export JWT_SECRET_KEY="your-secret-key"

# Run database migrations
alembic upgrade head

# Start development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend Development
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

## API Documentation

The FastAPI backend automatically generates interactive API documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Main API Endpoints

#### Authentication
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/refresh` - Refresh access token
- `POST /api/v1/auth/logout` - User logout

#### Portfolio Management
- `GET /api/v1/portfolio/list` - List user portfolios
- `POST /api/v1/portfolio/create` - Create new portfolio
- `POST /api/v1/portfolio/optimize` - Optimize portfolio weights
- `POST /api/v1/portfolio/backtest` - Run portfolio backtest

#### Market Regimes
- `POST /api/v1/regimes/identify` - Identify market regimes
- `POST /api/v1/regimes/forecast` - Forecast future regime
- `GET /api/v1/regimes/current` - Get current market regime
- `GET /api/v1/regimes/methods` - Available regime methods

#### Backtesting
- `POST /api/v1/backtesting/run` - Run comprehensive backtest
- `POST /api/v1/backtesting/multi-horizon` - Multi-horizon analysis
- `GET /api/v1/backtesting/{task_id}/results` - Get backtest results

#### Scenario Analysis
- `POST /api/v1/scenarios/create` - Create custom scenario
- `POST /api/v1/scenarios/stress-test` - Run stress test
- `POST /api/v1/scenarios/monte-carlo` - Monte Carlo simulation
- `POST /api/v1/scenarios/historical-replay` - Historical scenario replay

## Production Deployment

### Docker Compose Production
```bash
# Start with production profile
docker-compose --profile production up -d

# Scale workers
docker-compose --profile production up -d --scale worker=3
```

### Environment Variables

#### Backend (.env)
```env
DATABASE_URL=postgresql://user:password@db:5432/regimeshift
REDIS_URL=redis://redis:6379/0
JWT_SECRET_KEY=your-super-secret-jwt-key
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7
FRED_API_KEY=your-fred-api-key
```

#### Frontend (.env)
```env
VITE_API_BASE_URL=http://localhost:8000
VITE_APP_NAME="Dynamic Investment Strategies"
```

## Database Management

### Migrations
```bash
# Create new migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

### Backup and Restore
```bash
# Backup
docker-compose exec db pg_dump -U regimeshift regimeshift > backup.sql

# Restore
docker-compose exec -T db psql -U regimeshift regimeshift < backup.sql
```

## Testing

### Backend Tests
```bash
cd backend
pytest tests/ -v --cov=app
```

### Frontend Tests
```bash
cd frontend
npm run test
npm run test:coverage
```

### Integration Tests
```bash
# Start test environment
docker-compose -f docker-compose.test.yml up -d

# Run integration tests
pytest tests/integration/ -v
```

## Monitoring and Logging

### Application Logs
```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f backend
docker-compose logs -f frontend
```

### Health Checks
- Backend: http://localhost:8000/health
- Database: `docker-compose exec db pg_isready`
- Redis: `docker-compose exec redis redis-cli ping`

## Security

### Production Security Checklist
- [ ] Change default JWT secret key
- [ ] Use strong database passwords
- [ ] Enable HTTPS with SSL certificates
- [ ] Set up proper firewall rules
- [ ] Enable database encryption
- [ ] Configure backup encryption
- [ ] Set up monitoring and alerting
- [ ] Regular security updates

## Performance Optimization

### Backend Optimization
- Database connection pooling configured
- Redis caching for frequently accessed data
- Background task processing with Celery
- API response compression
- Database query optimization

### Frontend Optimization
- Code splitting and lazy loading
- Asset optimization and compression
- CDN integration ready
- Progressive Web App (PWA) ready
- Service worker for offline functionality

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue in the GitHub repository
- Check the API documentation at `/docs`
- Review the application logs for debugging

## Acknowledgments

- Built on top of the Dynamic Investment Strategies research
- Uses modern web development best practices
- Integrates with FRED economic data API
- Implements professional financial analysis methodologies