from sqlmodel import SQLModel, create_engine, Session

sqlite_file_name = "brain.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, echo=False, connect_args=connect_args)

# Initialize the database
def init_db():
    from .models import BrainDump, WeeklyPlan, Task  # ensure all tables exist
    SQLModel.metadata.create_all(engine)

# Get a session
def get_session():
    with Session(engine) as session:
        yield session
