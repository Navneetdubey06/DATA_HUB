import uuid
import time
import logging

logger = logging.getLogger(__name__)

class DashboardService:
    """Service for creating and managing dynamic dashboards"""

    def __init__(self):
        self.dashboard_store = {}  # Store dashboard configurations
        self.dashboard_timestamps = {}  # Track dashboard access times
        self.MAX_DASHBOARDS = 50
        self.DASHBOARD_TIMEOUT = 86400  # 24 hours

    def create_dashboard(self, df, session_id, title, widgets, layout):
        """Create a new dashboard"""
        try:
            # Validate session exists (this would be checked by the main app)
            # Generate unique dashboard ID
            dashboard_id = str(uuid.uuid4())
            logger.info(f"Generated dashboard ID: {dashboard_id} for session: {session_id}")

            # Validate widgets against available data
            validated_widgets = []
            for widget in widgets:
                widget_type = widget.get('type')
                if widget_type in ['histogram', 'scatter', 'line', 'bar', 'box', 'heatmap', 'pie']:
                    # Basic validation - could be enhanced
                    validated_widgets.append(widget)
                else:
                    logger.warning(f"Invalid widget type: {widget_type}")
                    return {'error': f'Unsupported widget type: {widget_type}'}

            # Store dashboard configuration
            dashboard_config = {
                'title': title,
                'session_id': session_id,
                'widgets': validated_widgets,
                'layout': layout,
                'created_at': time.time(),
                'dataset_shape': df.shape,
                'columns': list(df.columns)
            }

            self.dashboard_store[dashboard_id] = dashboard_config
            self.dashboard_timestamps[dashboard_id] = time.time()

            # Clean up old dashboards if needed
            self._cleanup_expired_dashboards()

            logger.info(f"Successfully created dashboard: {dashboard_id}")

            return {
                'dashboard_id': dashboard_id,
                'title': title,
                'widget_count': len(validated_widgets),
                'message': 'Dashboard created successfully'
            }

        except Exception as e:
            logger.error(f"Create dashboard error: {str(e)}")
            return {'error': 'Internal server error during dashboard creation'}

    def get_dashboard(self, dashboard_id, session_id):
        """Retrieve dashboard configuration"""
        try:
            # Validate dashboard exists
            if dashboard_id not in self.dashboard_store:
                logger.warning(f"Dashboard not found: {dashboard_id}")
                return {'error': 'Dashboard not found'}

            dashboard_config = self.dashboard_store[dashboard_id]

            # Check if associated session still exists
            if dashboard_config['session_id'] != session_id:
                logger.warning(f"Dashboard session mismatch for dashboard: {dashboard_id}")
                return {'error': 'Dashboard session mismatch'}

            self.dashboard_timestamps[dashboard_id] = time.time()  # Update access time

            logger.info(f"Retrieved dashboard: {dashboard_id}")

            return {
                'dashboard_id': dashboard_id,
                'title': dashboard_config['title'],
                'widgets': dashboard_config['widgets'],
                'layout': dashboard_config['layout'],
                'dataset_info': {
                    'shape': dashboard_config['dataset_shape'],
                    'columns': dashboard_config['columns']
                },
                'created_at': dashboard_config['created_at']
            }

        except Exception as e:
            logger.error(f"Get dashboard error: {str(e)}")
            return {'error': 'Internal server error during dashboard retrieval'}

    def update_dashboard(self, dashboard_id, session_id, title=None, widgets=None, layout=None):
        """Update an existing dashboard"""
        try:
            if dashboard_id not in self.dashboard_store:
                return {'error': 'Dashboard not found'}

            dashboard_config = self.dashboard_store[dashboard_id]

            if dashboard_config['session_id'] != session_id:
                return {'error': 'Dashboard session mismatch'}

            # Update fields if provided
            if title is not None:
                dashboard_config['title'] = title
            if widgets is not None:
                # Re-validate widgets
                validated_widgets = []
                for widget in widgets:
                    widget_type = widget.get('type')
                    if widget_type in ['histogram', 'scatter', 'line', 'bar', 'box', 'heatmap', 'pie']:
                        validated_widgets.append(widget)
                    else:
                        return {'error': f'Unsupported widget type: {widget_type}'}
                dashboard_config['widgets'] = validated_widgets
            if layout is not None:
                dashboard_config['layout'] = layout

            dashboard_config['updated_at'] = time.time()
            self.dashboard_timestamps[dashboard_id] = time.time()

            logger.info(f"Updated dashboard: {dashboard_id}")

            return {
                'dashboard_id': dashboard_id,
                'message': 'Dashboard updated successfully'
            }

        except Exception as e:
            logger.error(f"Update dashboard error: {str(e)}")
            return {'error': 'Internal server error during dashboard update'}

    def delete_dashboard(self, dashboard_id, session_id):
        """Delete a dashboard"""
        try:
            if dashboard_id not in self.dashboard_store:
                return {'error': 'Dashboard not found'}

            dashboard_config = self.dashboard_store[dashboard_id]

            if dashboard_config['session_id'] != session_id:
                return {'error': 'Dashboard session mismatch'}

            # Remove dashboard
            del self.dashboard_store[dashboard_id]
            del self.dashboard_timestamps[dashboard_id]

            logger.info(f"Deleted dashboard: {dashboard_id}")

            return {'message': 'Dashboard deleted successfully'}

        except Exception as e:
            logger.error(f"Delete dashboard error: {str(e)}")
            return {'error': 'Internal server error during dashboard deletion'}

    def list_dashboards(self, session_id):
        """List all dashboards for a session"""
        try:
            user_dashboards = []
            for dashboard_id, config in self.dashboard_store.items():
                if config['session_id'] == session_id:
                    user_dashboards.append({
                        'dashboard_id': dashboard_id,
                        'title': config['title'],
                        'created_at': config['created_at'],
                        'widget_count': len(config['widgets'])
                    })

            return {'dashboards': user_dashboards}

        except Exception as e:
            logger.error(f"List dashboards error: {str(e)}")
            return {'error': 'Internal server error during dashboard listing'}

    def _cleanup_expired_dashboards(self):
        """Clean up expired dashboards"""
        current_time = time.time()
        expired_dashboards = []

        for dashboard_id, timestamp in self.dashboard_timestamps.items():
            if current_time - timestamp > self.DASHBOARD_TIMEOUT:
                expired_dashboards.append(dashboard_id)

        for dashboard_id in expired_dashboards:
            if dashboard_id in self.dashboard_store:
                del self.dashboard_store[dashboard_id]
            if dashboard_id in self.dashboard_timestamps:
                del self.dashboard_timestamps[dashboard_id]
            logger.info(f"Cleaned up expired dashboard: {dashboard_id}")

        # Limit total dashboards
        if len(self.dashboard_store) > self.MAX_DASHBOARDS:
            sorted_dashboards = sorted(self.dashboard_timestamps.items(), key=lambda x: x[1])
            dashboards_to_remove = len(self.dashboard_store) - self.MAX_DASHBOARDS

            for dashboard_id, _ in sorted_dashboards[:dashboards_to_remove]:
                if dashboard_id in self.dashboard_store:
                    del self.dashboard_store[dashboard_id]
                if dashboard_id in self.dashboard_timestamps:
                    del self.dashboard_timestamps[dashboard_id]
                logger.info(f"Removed old dashboard due to limit: {dashboard_id}")

# Global instance
dashboard_service = DashboardService()