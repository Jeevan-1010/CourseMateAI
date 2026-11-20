from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database import get_engine
from backend.app.models import (
    Conversation,
    ConversationSummary,
    Message,
    StudyDocument,
)

def _session() -> Session:
    return Session(get_engine())


def save_document(filename: str, file_type: str, file_size_bytes: int) -> str:
    with _session() as session:
        document = session.scalar(
            select(StudyDocument)
            .where(StudyDocument.filename == filename)
            .order_by(StudyDocument.created_at.desc())
        )
        if document is None:
            document = StudyDocument(
                filename=filename,
                file_type=file_type,
                file_size_bytes=file_size_bytes,
                status="uploaded",
            )
            session.add(document)
        else:
            document.file_type = file_type
            document.file_size_bytes = file_size_bytes
            document.status = "uploaded"
        session.commit()
        session.refresh(document)
        return str(document.id)


def get_document_id(filename: str) -> str | None:
    with _session() as session:
        document = session.scalar(
            select(StudyDocument)
            .where(StudyDocument.filename == filename)
            .order_by(StudyDocument.created_at.desc())
        )
        return str(document.id) if document else None


def create_conversation(document_id: str, title: str | None = None) -> str:
    with _session() as session:
        conversation = Conversation(
            document_id=document_id,
            title=title or "New conversation",
        )
        session.add(conversation)
        session.commit()
        session.refresh(conversation)
        return str(conversation.id)


def conversation_belongs_to_document(
    conversation_id: str, document_id: str
) -> bool:
    with _session() as session:
        conversation = session.get(Conversation, conversation_id)
        return bool(
            conversation and str(conversation.document_id) == document_id
        )


def load_conversation_context(conversation_id: str, limit: int = 8):
    with _session() as session:
        conversation = session.get(Conversation, conversation_id)
        if conversation is None:
            return None

        summary_record = session.scalar(
            select(ConversationSummary).where(
                ConversationSummary.conversation_id == conversation_id
            )
        )
        messages = session.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        ).all()

        recent = [
            {"role": message.role, "content": message.content}
            for message in reversed(messages)
        ]
        summary = summary_record.summary if summary_record else ""
        return summary, recent


def save_exchange(
    conversation_id: str, question: str, answer: str
) -> None:
    with _session() as session:
        conversation = session.get(Conversation, conversation_id)
        if conversation is None:
            raise ValueError("Conversation does not exist.")

        session.add_all([
            Message(
                conversation_id=conversation_id,
                role="user",
                content=question,
            ),
            Message(
                conversation_id=conversation_id,
                role="assistant",
                content=answer,
            ),
        ])
        conversation.updated_at = __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        )
        session.commit()


def list_documents():
    with _session() as session:
        documents = session.scalars(
            select(StudyDocument).order_by(StudyDocument.created_at.desc())
        ).all()
        return [
            {
                "filename": item.filename,
                "file_type": item.file_type,
                "file_size_bytes": item.file_size_bytes,
                "status": item.status,
                "created_at": item.created_at.isoformat()
                if item.created_at else None,
            }
            for item in documents
        ]


def list_conversations(document_id: str):
    with _session() as session:
        conversations = session.scalars(
            select(Conversation)
            .where(Conversation.document_id == document_id)
            .order_by(Conversation.updated_at.desc())
        ).all()
        return [
            {
                "id": str(item.id),
                "title": item.title,
                "created_at": item.created_at.isoformat()
                if item.created_at else None,
                "updated_at": item.updated_at.isoformat()
                if item.updated_at else None,
            }
            for item in conversations
        ]


def get_conversation_messages(conversation_id: str):
    with _session() as session:
        conversation = session.get(Conversation, conversation_id)
        if conversation is None:
            return None
        messages = session.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
        ).all()
        return [
            {
                "role": message.role,
                "content": message.content,
                "created_at": message.created_at.isoformat()
                if message.created_at else None,
            }
            for message in messages
        ]
